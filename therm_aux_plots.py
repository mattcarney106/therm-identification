import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.segments import DataQualityFlag
from gwpy.time import Time, tconvert, to_gps, from_gps
from datetime import date, timedelta
from calibplot.calfetcher import CalFetcher
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
Script to read in auxiliary channel data
around thermalization times and plot the results.
'''

def get_days_between(start, end):

    days = []

    delta = end_date - start_date

    for i in range(delta.datetime.days + 1):
        # Get list of start times for each day
        day = start + timedelta(days=i)
        day = to_gps(day).gpsSeconds
        days.append(day)

    return days

IFO = 'H1'

# Set start and end date and get list of all days in between
start_date = Time('2023-06-17')
end_date = Time('2023-06-21')
day_starts_gps = get_days_between(start_date, end_date)

channel = IFO + ':ASC-X_TR_B_NSUM_OUTPUT'

laser_power = '75W'

lp_plot_range_dict = {'60W':{'ASC-X_TR_B_NSUM_OUTPUT': [3.8e5, 4.3e5], 'GDS-CALIB_KAPPA_C': [], 'GDS-CALIB_F_S_SQUARED': []},
                      '75W':{'ASC-X_TR_B_NSUM_OUTPUT': [4.3e5, 5e5], 'GDS-CALIB_KAPPA_C': [0.98, 1.1], 'GDS-CALIB_F_S_SQUARED': [-100, 40]}}

# Get segments
locked_times = DataQualityFlag.query(IFO + ':DMT-GRD_ISC_LOCK_NOMINAL:1',
                                     start_date,
                                     end_date)
# Determine if first lock stretch is just the start of a new day
for locked_time in locked_times.active:
    if locked_time[0].is_integer() and int(locked_time[0]) in day_starts_gps:
        print('Lock segment starting with:')
        print('GPS Time: {gps_start}'.format(gps_start=str(locked_time[0])))
        print('DateTime: {datetime_start}'.format(datetime_start=str(from_gps(locked_time[0]))))
        print('is just the beginning of a new day.')
        print('Removing from list of segments to analyze.')
        locked_times.active.remove(locked_time)


for segment in locked_times.active:
    # Fetch data for current time segment
    gps_start_time = segment[0]
    gps_end_time = segment[1]

    duration = gps_end_time - gps_start_time
    if duration <= 15*60:
        print('Lock segment duration of {dur} s is less than 15 minutes.'.format(dur=duration))

    print('Fetching and plotting data segment from {start} to {stop}'.format(start=gps_start_time, stop=gps_end_time))

    # Plot channels for time segment
    # Plot ASC-X thresholds for 75W power 
    # []    
    plot_channels = ['ASC-X_TR_B_NSUM_OUTPUT', 'GDS-CALIB_KAPPA_C', 'GDS-CALIB_F_S_SQUARED']
    channel_dict = {plot_channels[0]: {'thresholds': [4.03e5, 4.05e5, 4.1e5], 'color': 'blue'},
                    plot_channels[1]: {'thresholds':[], 'color': 'black'},
                    plot_channels[2]: {'thresholds':[0.], 'color': 'red'}}
    
    for plot_channel in plot_channels:
        try:
            channel_name = IFO + ':' + plot_channel
            data = TimeSeries.fetch(channel_name,
                                    gps_start_time,
                                    gps_end_time,
                                    allow_tape=True,
                                    verbose=True)

            # If we're looking at GDS-CALIB_F_S_SQUARED, throw out first 5 min of data
            if plot_channel == 'GDS-CALIB_F_S_SQUARED':
                # Find when first 5 minutes is over
                time_from_start = data.times.value - gps_start_time
                cutoff_time_min = 5
                cutoff_ind = np.where(time_from_start >= 5*60)[0][0]

                #print(time_from_start[:100])
                # Chop off first 5 minutes of data
                data = data[cutoff_ind:]


            # Determine where threshold is first met:
            thresholds = channel_dict[plot_channel]['thresholds']
            first_inds = [np.where(data.value >= threshold)[0][0] for threshold in thresholds if len(np.where(data.value >= threshold)[0]) != 0]
            threshold_times = [data.times.value[ind] for ind in first_inds if len(first_inds) != 0]

            print('Channel: ', plot_channel)
            print('Threshold times: ', threshold_times)

            # Store the threshol
            channel_dict[plot_channel]['threshold_times'] = threshold_times
            channel_plot = data.plot(ylabel=channel_name, xlim=[gps_start_time, gps_end_time], ylim=lp_plot_range_dict[laser_power][plot_channel])
            plot_ax = channel_plot.gca()

            # Get ylims and plot threshold lines
            yrange = plot_ax.get_ylim()
            for threshold_time in threshold_times:
                plot_ax.axvline(x=threshold_time, ymin=-100, ymax=yrange[1], linestyle='--', color=channel_dict[plot_channel]['color'])

            channel_plot_name = channel_name + '_' + str(gps_start_time) + '_' + str(gps_end_time) + '_time_series.jpg'

            # Make outdir
            home_dir = os.path.expanduser('~')
            outdir = home_dir + '/public_html/thermalization/' + str(start_date.datetime.date()) + '_' + str(end_date.datetime.date()) + '/' + str(gps_start_time) + '_' + str(gps_end_time) + '/'

            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            print(outdir + channel_plot_name)
            channel_plot.save(outdir+channel_plot_name)
            channel_plot.close()
        except Exception as error:
            print('Unable to fetch and plot {channel} data for current time range:'.format(channel=channel))
            print(error)

    # Plot transfer function magnitdue and phase
    config_file = '/home/matthew.carney/config/influx_config_LHO.ini'
    mag_fetcher = CalFetcher(config_file)
    phase_fetcher = CalFetcher(config_file)

    fields = ['time', 'data', 'oscillator_frequency', 'strain_channel', 'lock_state']
    phase_meas = ['TF_phase']
    mag_meas = ['TF_mag']
    conditions = [''' "line_on" = 'on"' ''', ''' "coh_threshold" = 'cohok' ''']

    phase_df = phase_fetcher.fetch_data(gps_start_time, gps_end_time, phase_meas, fields, conditions=conditions)
    mag_df = mag_fetcher.fetch_data(gps_start_time, gps_end_time, mag_meas, fields, conditions=conditions)

    lp_plot_freq_map = {'60W': [17.1, 104.23], '75W': [17.1, 102.13]}
    plot_freqs = lp_plot_freq_map[laser_power]
    mag_df_filtered = mag_df[mag_df['oscillator_frequency'].isin(plot_freqs)]
    phase_df_filtered = phase_df[phase_df['oscillator_frequency'].isin(plot_freqs)]

    mag_ts_dict = TimeSeriesDict()
    phase_ts_dict = TimeSeriesDict()
    for freq in plot_freqs:
        # Isolate values for specific frequency
        mag_freq_iso_df = mag_df_filtered[mag_df_filtered['oscillator_frequency'] == freq]
        phase_freq_iso_df = phase_df_filtered[phase_df_filtered['oscillator_frequency'] == freq]

        mag_ts = TimeSeries(np.array(mag_freq_iso_df['TF_mag'].values), times=np.array(mag_freq_iso_df['time'].values))
        phase_ts = TimeSeries(np.array(phase_freq_iso_df['TF_phase'].values), times=np.array(phase_freq_iso_df['time'].values))
        
        mag_ts_dict[freq] = mag_ts
        phase_ts_dict[freq] = phase_ts

    mag_fig = mag_ts_dict.plot(ylabel='TF_mag', ylim=[0.85, 1.15])
    phase_fig = phase_ts_dict.plot(ylabel='TF_phase', ylim=[-3, 4])

    mag_ax = mag_fig.gca()
    phase_ax = phase_fig.gca()

    # Plot vertical lines for all the thresholds
    for channel in channel_dict.keys():
        for time, threshold in zip(channel_dict[channel]['threshold_times'], channel_dict[channel]['thresholds']):
            mag_ax.axvline(time, -1000, 1000, color=channel_dict[channel]['color'],
                           label=channel + ' >= ' + str(threshold), linestyle='--')
            phase_ax.axvline(time, -1000, 1000, color=channel_dict[channel]['color'],
                             label=channel + ' >= ' + str(threshold), linestyle='--')

    mag_ax.legend(fontsize=10, loc='lower right')
    phase_ax.legend(fontsize=10, loc='lower right')

    mag_fig.save(outdir+'TF_mag_' + str(gps_start_time) + '_' + str(gps_end_time) + '_time_series.jpg')
    phase_fig.save(outdir+'TF_phase_' + str(gps_start_time) + '_' + str(gps_end_time) + '_time_series.jpg')


print('Successfully generated timeseries plots for channel ', channel)
