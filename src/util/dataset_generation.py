import os
import glob
import sys
import time
import json
import torch
from torch import nn
import torchvision
from torchvision.transforms import Compose
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import warnings

import util.transforms as T

import re
import numpy as np
import math
import random
from torch.utils.data import Dataset, random_split, WeightedRandomSampler

import util.misc as misc

import datetime
from collections import defaultdict
import pandas as pd
from scipy import interpolate

from scipy.ndimage import label as mark_overlap
from scipy.interpolate import griddata


class SpaceDataset(Dataset):
    def __init__(self, inputs, output, eh, token, shell_num, norm='', split_ratio=0.7, split_ratio_val=0.2, train='Train', new_split=False,
                 meta=None, predict=0):
        self.norm = norm
        self.eh = eh
        self.split_ratio = split_ratio
        self.split_ratio_val = split_ratio_val
        self.training = train
        self.window_step = 1  # if train else eh
        self.token = token
        self.shell_num = shell_num
        self.new_split = new_split
        self.predict = predict

        self.data, self.label, self.mask, self.label_mask, self.label_ind, self.meta_data, self.meta_data_mask = self.load_every(inputs, output,
                                                                                                                                 meta=meta)

    def load_every(self, input_dict, output, meta):
        if self.training == 'Train':
            start = 0
            end = math.ceil(output[0].shape[1] * self.split_ratio)
        elif self.training == 'Val':
            start = math.ceil(output[0].shape[1] * self.split_ratio)
            end = math.ceil(output[0].shape[1] * (self.split_ratio + self.split_ratio_val))
        else:
            start = math.ceil(output[0].shape[1] * (self.split_ratio + self.split_ratio_val))
            end = output[0].shape[1]
        if self.new_split:
            if self.training == 'Train':
                start = math.ceil(output[0].shape[1] * self.split_ratio_val)
                end = math.ceil(output[0].shape[1] * (self.split_ratio + self.split_ratio_val))
            elif self.training == 'Val':
                start = 0
                end = math.ceil(output[0].shape[1] * self.split_ratio_val)
        print(self.eh, self.window_step, start, end)

        data_list = []
        mask_list = []
        for sat_id, sat_data in input_dict.items():
            data = T.to_float_tensor(sat_data[0])[:, start:end]
            mask = T.to_float_tensor(sat_data[1])[:, start:end]
            staticis = sat_data[2]

            if self.norm == 'std':
                tem = (data - staticis['mean']) / staticis['std'] * (1 - mask) + (torch.zeros_like(data) + self.token) * mask
                data[:, :, self.shell_num:] = tem[:, :, self.shell_num:]
            elif self.norm == 'minmax':
                vmin, vmax = staticis['min'], staticis['max']
                data[:, :, self.shell_num:] = data[:, :, self.shell_num:] - vmin
                data[:, :, self.shell_num:] = data[:, :, self.shell_num:] / (vmax - vmin)
                tem = (data - 0.5) * 2 * (1 - mask) + (torch.zeros_like(data) + self.token) * mask
                data[:, :, self.shell_num:] = tem[:, :, self.shell_num:]

            data_list.append(data)
            mask_list.append(mask)

        data_list = torch.stack(data_list)  # Num_sat, C, H, W
        mask_list = torch.stack(mask_list)  # Num_sat, C, H, W
        if self.predict > 0:
            data_list = data_list[:, :, :-self.predict]
            mask_list = mask_list[:, :, :-self.predict]
        data = self.split_along_time(data_list)  # [H-eh+1, Num_sat, C, eh, W]
        mask = self.split_along_time(mask_list)  # [H-eh+1, Num_sat, C, eh, W]

        label = T.to_float_tensor(output[0])[:, start:end, self.shell_num:]
        label_mask = T.to_float_tensor(output[1])[:, start:end, self.shell_num:]
        label_ind = T.to_float_tensor(output[0])[:, start:end, :self.shell_num]
        staticis = output[2]
        if self.norm == 'std':
            label = (label - staticis['mean']) / staticis['std'] * (1 - label_mask) + (torch.zeros_like(label) + self.token) * label_mask
        elif self.norm == 'minmax':
            vmin, vmax = staticis['min'], staticis['max']
            label = label - vmin
            label = label / (vmax - vmin)
            label = (label - 0.5) * 2 * (1 - label_mask) + (torch.zeros_like(label) + self.token) * label_mask
        if self.predict > 0:
            label = label[:, self.predict:]
            label_mask = label_mask[:, self.predict:]
            label_ind = label_ind[:, self.predict:]
        label = self.label_split_along_time(label)  # [H-eh+1, C, eh, W]
        label_mask = self.label_split_along_time(label_mask)  # [H-eh+1, C, eh, W]
        label_ind = self.label_split_along_time(label_ind)  # [H-eh+1, C, eh, W]

        assert label.shape[0] == data.shape[0]

        meta_data = T.to_float_tensor(meta[0])[:, start:end, :]
        meta_data_mask = T.to_float_tensor(meta[1])[:, start:end, :]
        meta_staticis = meta[2]
        if self.predict > 0:
            meta_data = meta_data[:, :-self.predict]
            meta_data_mask = meta_data_mask[:, :-self.predict]
        meta_data = self.label_split_along_time(meta_data)  # [H-eh+1, C, eh, W]
        meta_data_mask = self.label_split_along_time(meta_data_mask)  # [H-eh+1, C, eh, W]

        assert meta_data.shape[0] == data.shape[0]

        return data, label, mask, label_mask, label_ind, meta_data, meta_data_mask

    def split_along_time(self, x):
        x = x.unfold(2, self.eh, self.window_step)
        x = x.permute(2, 0, 1, 4, 3)
        return x

    def label_split_along_time(self, x):
        x = x.unfold(1, self.eh, self.window_step)
        x = x.permute(1, 0, 3, 2)
        return x

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.mask[idx], self.label_mask[idx], self.label_ind[idx], self.meta_data[idx], self.meta_data_mask[idx]

    def __len__(self):
        return self.data.shape[0]


def make_balanced_sampler(targets, staticis, num_bins):
    # Bin the target variable
    bins = np.linspace(np.nanmin(targets.numpy()) - 0.01, np.nanmax(targets.numpy()) + 0.01, num_bins)
    y_binned = np.digitize(np.nanmax(targets.numpy(), axis=(-1, -2, -3)), bins)

    # Count the number of instances in each bin
    bin_counts = np.bincount(y_binned)

    # Define weights as the inverse of the bin counts
    weights = 1. / bin_counts[1:]
    weights = np.insert(weights, 0, 0)
    print(bins)
    print(f'Bins: {bin_counts}')
    print(f'Weights: {weights}')

    # Assign each instance the weight of its bin
    sample_weights = weights[y_binned]

    # Use the weights to create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return sampler


def smooth_image(y):
    a = np.arange(y.shape[1])
    b = np.arange(y.shape[0])
    X, Y = np.meshgrid(a, b)

    # Step 2: Flatten the data and filter out NaN values
    known_points = ~np.isnan(y)
    known_values = y[known_points]
    known_coordinates = np.vstack((X[known_points], Y[known_points])).T

    # Step 3: Interpolate using griddata
    means_y = griddata(known_coordinates, known_values, (X, Y), method='linear', fill_value=np.nan)
    return means_y


def calculate_mean_ratio(array1, array2):
    # Masks where values are not nan for both arrays
    not_nan1 = ~np.isnan(array1)
    not_nan2 = ~np.isnan(array2)

    # Overlapping mask
    overlap = np.logical_and(not_nan1, not_nan2)

    # Label connected regions of overlap
    labeled_overlap, num_features = mark_overlap(overlap)

    ratios = []
    for i in range(1, num_features + 1):
        current_region = (labeled_overlap == i)

        overlap_values1 = array1[current_region]
        overlap_values2 = array2[current_region]

        # Calculate the ratio of means for the current region
        ratio = np.mean(overlap_values1) / np.mean(overlap_values2)
        ratios.append(ratio)

    return np.mean(ratios)


def deal_str_date(str_date, start_date='20130220'):
    str_date = str(str_date)
    start = datetime.datetime(int(start_date[:4]), int(start_date[4:6]),
                              int(start_date[6:]))
    day_num = (datetime.datetime(int(str_date[:4]), int(str_date[4:6]),
                                 int(str_date[6:])) - start).days
    return day_num


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def creat_dataset(args, rand=True):
    split_ratio = 0.7
    split_ratio_val = 0.2

    print('Loading data')
    name = args.anno_path
    inputs, output, shell_num = load_data(args, split_ratio)
    meta = load_meta(args, split_ratio)
    geo_all = load_geo(args)

    en, dn = args.embed_dim, args.latent_dim

    num_sat = len(inputs)

    arbitrary_value = next(iter(inputs.values()))

    ed, ew = arbitrary_value[0].shape[0], arbitrary_value[0].shape[-1]
    eh = args.input_size
    dd, dh, dw = 1, eh, output[0].shape[-1] - shell_num

    meta_dim = meta[0].shape[-1] - shell_num

    print('Loading training data')
    dataset_train = SpaceDataset(
        inputs, output, eh, args.token, shell_num, norm=args.norm, split_ratio=split_ratio, split_ratio_val=split_ratio_val, train='Train',
        new_split=args.new_split, meta=meta, predict=args.predict)

    print('Loading validation data')

    dataset_valid = SpaceDataset(
        inputs, output, eh, args.token, shell_num, norm=args.norm, split_ratio=split_ratio, split_ratio_val=split_ratio_val, train='Val',
        new_split=args.new_split, meta=meta, predict=args.predict)

    print('Loading testing data')

    dataset_test = SpaceDataset(
        inputs, output, eh, args.token, shell_num, norm=args.norm, split_ratio=split_ratio, split_ratio_val=split_ratio_val, train='Test',
        new_split=args.new_split, meta=meta, predict=args.predict)

    print(name, len(dataset_train), len(dataset_valid), len(dataset_test))

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        train_sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=rand
        )
        print("Sampler_train = %s" % str(train_sampler))

        valid_sampler = torch.utils.data.DistributedSampler(
            dataset_valid, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        test_sampler = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        if rand:
            if not args.resample:
                train_sampler = torch.utils.data.RandomSampler(dataset_train)
            else:
                train_sampler = make_balanced_sampler(dataset_train.label, output[2], num_bins=10)
        else:
            train_sampler = SequentialSampler(dataset_train)
        valid_sampler = SequentialSampler(dataset_valid)
        test_sampler = SequentialSampler(dataset_test)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=default_collate, persistent_workers=True)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=default_collate, persistent_workers=True)
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=default_collate, persistent_workers=True)

    if args.norm == 'std':
        scaler1, scaler2 = output[2]['mean'], output[2]['std']
    elif args.norm == 'minmax':
        scaler1, scaler2 = output[2]['min'], output[2]['max']
    else:
        scaler1, scaler2 = None, None

    return en, ed, eh, ew, dn, dd, dh, dw, meta_dim, num_sat, shell_num, scaler1, scaler2, output[
        2], dataloader_train, dataloader_valid, dataloader_test, output, meta, geo_all


def load_data(args, split_ratio):
    file_path = args.anno_path
    txt_files = glob.glob(os.path.join(file_path, '*.txt'))

    main_input = defaultdict(list)
    target_list = []
    other_input_list = {}
    other_target_list = []

    fluxe_num = {'2000keV': 1, '120keV': 4, '1000keV': 2, '300keV': 3, 'above2MeV': 0, }
    token = args.token
    interp = args.interp
    omni_column = ['date_num', 'UT'] + ['Dst']  # ['Dst', 'S/W_speed', 'S/W_p+_den', 'S/W_pressure']

    for txt_file in txt_files:
        file_name = os.path.basename(txt_file).split('_')
        if file_name[0] == 'Input':
            if file_name[1] == 'GPS':
                SatID = file_name[2]
                if SatID not in args.choose_sat:
                    continue
                fluxe = file_name[5]
                start_date = file_name[6].split('-')[0]
                end_date = file_name[6].split('-')[1]
                if fluxe_num[fluxe] not in args.drop_fluxe:
                    main_input[SatID].append(
                        {'fluxe': fluxe_num[fluxe], 'start': start_date, 'end': end_date, 'file': txt_file})
            elif 'LANL' in file_name[1]:
                SatID = file_name[1]
                fluxe = file_name[4]
                start_date = file_name[5].split('-')[0]
                end_date = file_name[5].split('-')[1]
                other_target_list.append({'fluxe': fluxe_num[fluxe], 'start': start_date, 'end': end_date, 'file': txt_file})
            elif file_name[2] == 'OMNI' and args.omni:
                SatID = file_name[2]
                fluxe = None
                start_date = file_name[3].split('-')[0]
                end_date = file_name[3].split('-')[1][:-4]
                other_input_list['OMNI'] = {'fluxe': 'SolarWind', 'start': start_date, 'end': end_date, 'file': txt_file}
            else:
                continue
        elif file_name[0] == 'Target':
            SatID = file_name[1]
            fluxe = file_name[4]
            start_date = file_name[5].split('-')[0]
            end_date = file_name[5].split('-')[1]
            target_list.append({'fluxe': fluxe_num[fluxe], 'start': start_date, 'end': end_date, 'file': txt_file})

    input = {}
    if not args.low_shell:
        column = ['date_num', 'UT'] + ["{:.3f}".format(i) for i in np.arange(2.8, 8.2, 0.2)]
    else:
        column = ['date_num', 'UT'] + ["{:.3f}".format(i) for i in np.arange(2.8, 5.1, 0.2)]
    shell_num = 2

    for satid, sat_data in main_input.items():
        c, h, w = len(sat_data), (deal_str_date(sat_data[0]['end'], sat_data[0]['start']) + 1) * 24, len(column)
        if args.time_avg > 0:
            h = (h // args.time_avg)
        start = 0
        end = math.ceil(h * split_ratio)

        data_array = np.zeros((c, h, w), dtype='float')
        mask_array = np.zeros((c, h, w), dtype='bool')

        tem = np.full((len(sat_data), 1, 1), token, dtype='float')
        data_satistic = {'min': tem.copy(), 'max': tem.copy(), 'mean': tem.copy(), 'std': tem.copy()}

        sat_data.sort(key=lambda x: x['fluxe'])
        for i, data_dict in enumerate(sat_data):
            path = data_dict['file']

            df = pd.read_csv(path, delim_whitespace=True, skiprows=11)
            df['date_num'] = df['Date'].apply(deal_str_date)
            df = df[column]

            image = df.to_numpy()
            if args.low_shell:
                image = image[:, :14]
            if args.remove_shell > 0:
                image[:, args.remove_shell + shell_num] = np.nan
            if args.time_avg > 0:
                tlong, W = image.shape
                # Number of rows after dropping remainder
                new_T = (tlong // args.time_avg) * args.time_avg
                reshaped_data = image[:new_T].reshape(-1, args.time_avg, W)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image = np.nanmean(reshaped_data, axis=1)

            mask_nan = np.isnan(image)
            mask_valid = np.logical_not(mask_nan)

            if args.log:
                image[:, shell_num:] = T.log_transform(image[:, shell_num:], log10=args.log10)

            data_satistic['min'][i, 0, 0] = np.nanmin(image[start:end, shell_num:])
            data_satistic['max'][i, 0, 0] = np.nanmax(image[start:end, shell_num:])
            data_satistic['mean'][i, 0, 0] = np.nanmean(image[start:end, shell_num:])
            data_satistic['std'][i, 0, 0] = np.nanstd(image[start:end, shell_num:])

            if interp:
                for method in ['cubic', 'nearest']:
                    rest_of_image = image[:, shell_num:]
                    rest_mask_nan = np.isnan(rest_of_image)
                    rest_mask_valid = np.logical_not(rest_mask_nan)
                    coordinates_valid = np.array(np.nonzero(rest_mask_valid)).T
                    values_valid = rest_of_image[rest_mask_valid]
                    coordinates_nan = np.array(np.nonzero(rest_mask_nan)).T
                    image_interpolated = interpolate.griddata(coordinates_valid, values_valid, coordinates_nan,
                                                              method=method)
                    rest_of_image[rest_mask_nan] = image_interpolated
                    image[:, 2:] = rest_of_image
            else:
                image[mask_nan] = token

            data_array[i] = image
            mask_array[i] = mask_nan

        if args.omni:
            path = other_input_list['OMNI']['file']
            df = pd.read_csv(path, delim_whitespace=True, skiprows=10)
            df['date_num'] = df['Date'].apply(deal_str_date)
            df = df[omni_column]
            omni = df.to_numpy()

            repeated_first_two = np.tile(omni[:, :2][np.newaxis, :, :], (len(omni_column) - 2, 1, 1))
            rest_expanded = np.tile(omni[:, 2:].T[:, :, np.newaxis], (1, 1, w - 2))
            omni = np.concatenate([repeated_first_two, rest_expanded], axis=2)

            if args.time_avg > 0:
                C, tlong, W = omni.shape
                print(f'OMNI Shape: {omni.shape}')
                new_T = (tlong // args.time_avg) * args.time_avg
                reshaped_data = omni[:, :new_T].reshape(1, -1, args.time_avg, W)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    omni = np.nanmean(reshaped_data, axis=2)

            for i in range(omni.shape[0]):
                tem_min = np.nanmin(omni[i, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['min'] = np.concatenate([data_satistic['min'], tem_min], axis=0)
                tem_max = np.nanmax(omni[i, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['max'] = np.concatenate([data_satistic['max'], tem_max], axis=0)

                tem_mean = (tem_max + tem_min) / 2  # np.nanmean(omni[i, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['mean'] = np.concatenate([data_satistic['mean'], tem_mean], axis=0)
                tem_std = (tem_max - tem_min) / 2  # np.nanstd(omni[i, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['std'] = np.concatenate([data_satistic['std'], tem_std], axis=0)

            sat_mask = np.all(mask_array == 1, axis=0)[np.newaxis,]
            omni_mask = np.isnan(omni)
            omni[omni_mask] = token
            n = omni_mask.shape[0]
            omni_mask = np.concatenate([omni_mask, sat_mask], axis=0)
            omni_mask = ~np.all(omni_mask == 0, axis=0).astype(bool)
            omni_mask = np.repeat(omni_mask[np.newaxis,], n, axis=0)

            data_array = np.concatenate([data_array, omni], axis=0)
            mask_array = np.concatenate([mask_array, omni_mask], axis=0)

        if args.input_geo:
            path = other_target_list[0]['file']
            df = pd.read_csv(path, delim_whitespace=True, skiprows=11)
            df['date_num'] = df['Date'].apply(deal_str_date)
            df = df[column]
            geo_image = df.to_numpy()

            if args.time_avg > 0:
                tlong, W = geo_image.shape
                new_T = (tlong // args.time_avg) * args.time_avg
                reshaped_data = geo_image[:new_T].reshape(-1, args.time_avg, W)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    geo_image = np.nanmean(reshaped_data, axis=1)

            geo_image = geo_image[np.newaxis, ]
            geo_mask = np.isnan(geo_image)

            if args.log and args.geo_log and int(args.ckpt_name[4:]) >= 101 and not args.old_geo_input:
                geo_image[:, :, shell_num:] = T.log_transform(geo_image[:, :, shell_num:], log10=args.log10)
            else:
                image[:, shell_num:] = T.log_transform(image[:, shell_num:], log10=args.log10)

            if not args.old_geo_input:
                tem = np.nanmin(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['min'] = np.concatenate([data_satistic['min'], tem], axis=0)
                tem = np.nanmax(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['max'] = np.concatenate([data_satistic['max'], tem], axis=0)
                tem = np.nanmean(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['mean'] = np.concatenate([data_satistic['mean'], tem], axis=0)
                tem = np.nanstd(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['std'] = np.concatenate([data_satistic['std'], tem], axis=0)

                geo_image[geo_mask] = token
                data_array = np.concatenate([data_array, geo_image], axis=0)
                mask_array = np.concatenate([mask_array, geo_mask], axis=0)
            else:
                geo_image[geo_mask] = token
                data_array = np.concatenate([data_array, geo_image], axis=0)
                mask_array = np.concatenate([mask_array, geo_mask], axis=0)

                tem = np.nanmin(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['min'] = np.concatenate([data_satistic['min'], tem], axis=0)
                tem = np.nanmax(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['max'] = np.concatenate([data_satistic['max'], tem], axis=0)
                tem = np.nanmean(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['mean'] = np.concatenate([data_satistic['mean'], tem], axis=0)
                tem = np.nanstd(geo_image[:, start:end, shell_num:]).reshape(1, 1, 1)
                data_satistic['std'] = np.concatenate([data_satistic['std'], tem], axis=0)

        input[satid] = [data_array, mask_array, data_satistic]

    # c, h, w = 1, (deal_str_date(target_list[0]['end'], target_list[0]['start']) + 1) * 24, len(column)
    output = []
    path = target_list[0]['file']
    df = pd.read_csv(path, delim_whitespace=True, skiprows=11)
    df['date_num'] = df['Date'].apply(deal_str_date)
    df = df[column]
    image = df.to_numpy()
    if args.geo:
        path = other_target_list[0]['file']
        df = pd.read_csv(path, delim_whitespace=True, skiprows=11)
        df['date_num'] = df['Date'].apply(deal_str_date)
        df = df[column]
        geo_image = df.to_numpy()

        if args.old_version:
            # Simplest method to combine GEO
            image = np.where(~np.isnan(image), image, geo_image)
        else:
            # New method to combine GEO
            r = 1 # calculate_mean_ratio(image, geo_image)
            image = np.concatenate([image[:, :17 + shell_num], geo_image[:, shell_num + 17:] * r], axis=1)
    if args.smooth_label:
        image = smooth_image(image)
    if args.time_avg > 0:
        tlong, W = image.shape
        new_T = (tlong // args.time_avg) * args.time_avg
        image = image[:new_T].reshape(-1, args.time_avg, W)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = np.nanmean(image, axis=1)
    mask_nan = np.isnan(image)
    image[:, shell_num:] = T.log_transform(image[:, shell_num:], log10=args.log10) if args.log else image[:, shell_num:]
    label_satistic = {'min': np.nanmin(image[:, shell_num:]), 'max': np.nanmax(image[:, shell_num:]), 'mean': np.nanmean(image[:, shell_num:]),
                      'std': np.nanstd(image[:, shell_num:])}

    image[mask_nan] = token
    output += [image[np.newaxis], mask_nan[np.newaxis], label_satistic]

    return input, output, shell_num


def load_meta(args, split_ratio):
    file_path = args.anno_path
    txt_files = glob.glob(os.path.join(file_path, '*.txt'))

    meta_list = {}

    token = 0
    interp = args.interp

    for txt_file in txt_files:
        file_name = os.path.basename(txt_file).split('_')
        if file_name[0] == 'Input':
            if file_name[2] == 'OMNI':
                SatID = file_name[2]
                fluxe = None
                start_date = file_name[3].split('-')[0]
                end_date = file_name[3].split('-')[1][:-4]
                meta_list['OMNI'] = {'fluxe': 'SolarWind', 'start': start_date, 'end': end_date, 'file': txt_file}
            elif '_'.join(file_name[2:6]) == 'lower_bound_fromPOES_P6' and file_name[6] == '3hr':
                SatID = file_name[2]
                fluxe = None
                start_date = file_name[7].split('-')[0]
                end_date = file_name[7].split('-')[1]
                meta_list['lower_bound_3'] = {'fluxe': 'lower_bound_3', 'start': start_date, 'end': end_date, 'file': txt_file}
            elif '_'.join(file_name[2:6]) == 'lower_bound_fromPOES_P6' and file_name[6] == '5hr':
                SatID = file_name[2]
                fluxe = None
                start_date = file_name[7].split('-')[0]
                end_date = file_name[7].split('-')[1]
                meta_list['lower_bound_5'] = {'fluxe': 'lower_bound_5', 'start': start_date, 'end': end_date, 'file': txt_file}
            else:
                continue

    omni_column = ['date_num', 'UT'] + args.meta_mode  # ['Dst', 'S/W_speed', 'S/W_p+_den', 'S/W_pressure']
    shell_num = 2

    h, w = (deal_str_date(meta_list['OMNI']['end'], meta_list['OMNI']['start']) + 1) * 24, len(omni_column) + 1 - shell_num
    start = 0
    end = math.ceil(h * split_ratio)
    data_satistic = np.full((4, w), np.nan, dtype='float')

    path = meta_list['OMNI']['file']
    df = pd.read_csv(path, delim_whitespace=True, skiprows=10)
    df['date_num'] = df['Date'].apply(deal_str_date)
    df = df[omni_column]
    omni = df.to_numpy()

    if args.time_avg > 0:
        tlong, W = omni.shape
        print(f'OMNI Shape: {omni.shape}')
        new_T = (tlong // args.time_avg) * args.time_avg
        reshaped_data = omni[:new_T].reshape(-1, args.time_avg, W)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            omni = np.nanmean(reshaped_data, axis=1)

    if args.meta_mode:
        data_satistic[0, :-1] = np.nanmin(omni[start:end, shell_num:])
        data_satistic[1, :-1] = np.nanmax(omni[start:end, shell_num:])
        data_satistic[2, :-1] = np.nanmean(omni[start:end, shell_num:])
        data_satistic[3, :-1] = np.nanstd(omni[start:end, shell_num:])

        for j in range(len(omni_column) - shell_num):
            omni[:, shell_num + j] = (omni[:, shell_num + j] - data_satistic[0, j]) / (data_satistic[1, j] - data_satistic[0, j])

    omni_mask = np.isnan(omni)
    omni[omni_mask] = token

    if args.meta_pose:
        if args.time_avg % 3 == 0:
            lower_bound_time = '3hr'
            lower_bound_key = 'lower_bound_3'
            args.time_avg = args.time_avg // 3
        elif args.time_avg % 5 == 0:
            lower_bound_time = '5hr'
            lower_bound_key = 'lower_bound_5'
            args.time_avg = args.time_avg // 5
        else:
            lower_bound_time = '3hr_pad'
            lower_bound_key = 'lower_bound_3'

        path = meta_list[lower_bound_key]['file']
        column = ['Lbound']
        df = pd.read_csv(path, delim_whitespace=True, skiprows=9)
        df = df[column]
        lbound = df.to_numpy()

        if lower_bound_time == '3hr_pad':
            lbound = np.repeat(lbound, 3, axis=0)

        if args.time_avg > 0:
            tlong, W = lbound.shape
            print(f'Lbound Shape: {lbound.shape}')
            new_T = (tlong // args.time_avg) * args.time_avg
            reshaped_data = lbound[:new_T].reshape(-1, args.time_avg, W)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lbound = np.nanmean(reshaped_data, axis=1)

        data_satistic[0, -1] = np.nanmin(lbound[start:end, 0])
        data_satistic[1, -1] = np.nanmax(lbound[start:end, 0])
        data_satistic[2, -1] = np.nanmean(lbound[start:end, 0])
        data_satistic[3, -1] = np.nanstd(lbound[start:end, 0])

        lbound_mask = np.isnan(lbound)
        lbound[lbound_mask] = token

        data = np.concatenate([omni, lbound], axis=-1)
        mask = np.concatenate([omni_mask, lbound_mask], axis=-1)
    else:
        data = omni
        mask = omni_mask

    output = [data[np.newaxis], mask[np.newaxis], data_satistic[np.newaxis]]

    return output


def load_geo(args):
    file_path = args.anno_path
    txt_files = glob.glob(os.path.join(file_path, '*.txt'))

    target_list = []

    fluxe_num = {'2000keV': 1, '120keV': 4, '1000keV': 2, '300keV': 3, 'above2MeV': 0, }
    token = args.token
    interp = args.interp
    column = ['date_num', 'UT'] + ["{:.3f}".format(i) for i in np.arange(2.8, 8.2, 0.2)]

    for txt_file in txt_files:
        file_name = os.path.basename(txt_file).split('_')
        if 'LANL' in file_name[1]:
            SatID = file_name[1]
            fluxe = file_name[4]
            start_date = file_name[5].split('-')[0]
            end_date = file_name[5].split('-')[1]
            target_list.append({'fluxe': fluxe_num[fluxe], 'start': start_date, 'end': end_date, 'file': txt_file})

    c, h, w = 1, (deal_str_date(target_list[0]['end'], target_list[0]['start']) + 1) * 24, len(column)
    output = []
    path = target_list[0]['file']
    df = pd.read_csv(path, delim_whitespace=True, skiprows=11)
    df['date_num'] = df['Date'].apply(deal_str_date)
    df = df[column]
    image = df.to_numpy()
    mask_nan = np.isnan(image)
    image[:, 2:] = T.log_transform(image[:, 2:]) if args.log else image[:, 2:]
    label_satistic = {'min': np.nanmin(image[:, 2:]), 'max': np.nanmax(image[:, 2:]), 'mean': np.nanmean(image[:, 2:]),
                      'std': np.nanstd(image[:, 2:])}

    image[mask_nan] = token
    output += [image[np.newaxis], mask_nan[np.newaxis], label_satistic]

    return output


if __name__ == '__main__':
    creat_dataset()
