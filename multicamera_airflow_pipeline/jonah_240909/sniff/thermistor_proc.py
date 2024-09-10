import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging
import sys
import cv2

import pynwb as nwb
import sniffing
import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")


def to_snake_case(string):
    string = string.strip('\n\r\t ')
    string = string.replace('-', '_')
    string = string.replace(' ', '_')
    string = string.lower()
    return string


class ThermistorProcessor:
    def __init__(
        self,
        recording_directory,
        output_directory,
        thermistor_file_pattern="*.txt",  # TODO: if becomes necessary, could pass this as a param thru google sheet or thru the airlfow config
        thermistor_file_exclude_patterns=["xxx", "metadata"],
        starting_sniff_min_prominence = 0.1,
        fs_interp=500,
        remove_low_amp_breaths=True,
        force_remove_low_amp_breaths=False,
        recompute_completed=False,
    ):
        """
        Parameters
        ----------
        recording_directory : Path
            Path to the directory containing the recording data.
        output_location : Path
            Path to the directory where the processed thermistor data will be saved.
        thermistor_file_pattern : str
            Pattern to match the thermistor file in the recording directory.
        thermistor_file_exclude_patterns : list
            List of patterns to exclude from the thermistor file search.
        starting_sniff_min_prominence : float
            Minimum prominence for the starting sniff detection.
        fs_interp : int
            Sampling frequency to be used for making the interpolated data.
        remove_low_amp_breaths : bool
            Whether to remove low amplitude breaths via detection of bi-modality in log-transformed breath amplitude distribution.
        force_remove_low_amp_breaths : bool
            Whether to force removal of low amplitude breaths even if there are more than X% of them.
        recompute_completed : bool
            Whether to recompute the processed thermistor data even if it already exists.
        
        
        """
        self.recording_directory = Path(recording_directory)
        self.output_directory = Path(output_directory)
        self.remove_low_amp_breaths = remove_low_amp_breaths
        self.force_remove_low_amp_breaths = force_remove_low_amp_breaths
        self.recompute_completed = recompute_completed


        # Validate only one thermistor file
        # thermistor_files = list(self.recording_directory.glob(thermistor_file_pattern))
        # thermistor_files = [
        #     i for i in thermistor_files if not any(exclude in i.stem for exclude in thermistor_file_exclude_patterns)
        # ]
        # if len(thermistor_files) == 0:
        #     raise FileNotFoundError("No thermistor files found in the recording directory.")
        # elif len(thermistor_files) > 1:
        #     raise ValueError("Multiple thermistor files found in the recording directory.")
        # else:
        #     pass

        # Get session metadata
        self.session_name = self.recording_directory.name
        self.nwb_filename = self.output_directory / f"{self.session_name}.nwb"

        # Set the validation directory
        if self.remove_low_amp_breaths:
            self.validation_dir_name = "validation_removeLowAmp"
        elif self.force_remove_low_amp_breaths:
            if not self.remove_low_amp_breaths:
                logger.warning("force_remove_low_amp_breaths is set to True, but remove_low_amp_breaths is set to False. Setting remove_low_amp_breaths to True.")
                self.remove_low_amp_breaths = True
            self.validation_dir_name = "validation_forceRemoveLowAmp"
        else:
            self.validation_dir_name = "validation_noRemoveLowAmp"

    def make_new_nwb_file(self):
        
        # Create the new NWB if needed
        if self.recompute_completed:
            mode = 'w'
        else:
            mode = 'x'
        
        # Get session start time (sort of fuding this)
        start_time = datetime.now()

        # Create the file
        nwbfile = nwb.NWBFile(
            session_description="Sniffing data",
            identifier=self.session,
            session_start_time=start_time)
        with nwb.NWBHDF5IO(self.nwb_filename, mode) as io:
            io.write(nwbfile)
        logger.info(f'Created {self.nwb_filename}')
    
    def run_thermistor_analysis(self):
        
        try:
            
            # Check if need to do anything
            with sniffing.nwb.io.open_nwb(self.nwb_filename, mode="r") as nwb_file:
                idi = sniffing.arduino.io.get_interpolated_data_interface(nwb_file, self.fs_interp)  # returns None if doesn't exist
                if (idi is not None) and ("instant_sniff_rate" in idi.time_series):
                    logger.info('Looks like file already has interpolated data and sniff data, skipping...')
                    do_sniff_flag = False
                else:
                    do_sniff_flag = True
    
            # Process sniffing if present
            if do_sniff_flag:
                
                # Load raw arduino data
                sniffing.arduino.io.load_arduino_data(self.nwb_filename, self.recording_directory, exclude_patterns=self.thermistor_file_exclude_patterns)

                # Interpolate it
                sniffing.arduino.io.interpolate_arduino_streams(self.nwb_filename, fs_interp=self.fs_interp)

                # Do sniff analyses
                sniffing.arduino.io.find_sniff_peaks_search(self.nwb_filename, fs_interp=self.fs_interp, secondary_peak_search=True)
                sniffing.arduino.io.process_breaths_for_nwb(self.nwb_filename, fs_interp=self.fs_interp, remove_low_amp_breaths=self.remove_low_amp_breaths, force_remove_low_amp=self.force_remove_low_amp) 
                sniffing.arduino.io.calculate_sniff_rates(self.nwb_filename, fs_interp=self.fs_interp)
                
            logger.info('Finished processing sniffing data')
        
        except BlockingIOError:
            logger.warning('Session is i/o blocked, NWB file may be open elsewhere or corrupted.')

    def load_odor_data_if_present(self):

        metadata_pattern = '*_metadata.yml'
        # Check if odor channels are present in the arduino data
        with sniffing.nwb.io.open_nwb(self.nwb_filename, mode="r") as nwb_file:
            if any(['odor' in k for k in nwb_file.acquisition.keys()]):
                pass
            else:
                return
        
        logger.info('Adding odor data...')

        # Find metadata file (eg what odors, concentrations, flow rates were used)
        odor_metadata = self.recording_directory.glob(metadata_pattern)
        if len(odor_metadata) == 0:
            logger.warning('No odor metadata found, skipping odor processing')
            return
        elif len(odor_metadata) > 1:
            logger.warning('Multiple odor metadata files found, skipping odor processing')
            return
        
        # Load metadata
        odor_dict = {}
        with open(odor_metadata, 'r') as f:
            metadata = yaml.load(f, yaml.Loader)
            for odor,val in metadata['odors'].items():
                odor_name = to_snake_case(odor).replace("_", "")
                odor_dict[odor_name] = val
        session_type = metadata['session']['session_type']
        if ('nothing' in session_type) or (session_type == "control (central only)"):
            logger.info('session labeled as "nothing" type (ie no clicks), skipping odor proc')
            return
        sniffing.arduino.io.process_odor_stims(self.nwb_filename, odor_dict)
            
            
    def make_validation_plots(self):
        sniffing.thermistor.viz.plot_thermistor_validations(self.nwb_filename, overwrite=False, out_dir=self.validation_dir_name)

    def check_completed(self):
        full_validation_path = (self.output_directory / self.validation_dir_name)
        if self.nwb_filename.exists() and full_validation_path.exists() and len(list(full_validation_path.glob("*.png"))) > 0:
            return True
        else:
            return False

    
    def run(self):

        # check if already completed
        if not self.recompute_completed:
            if self.check_completed():
                logger.info("Thermistor processing already completed")
                return

       # Run the analysis steps
        self.make_new_nwb_file()
        self.run_thermistor_analysis()
        self.load_odor_data_if_present()
        self.make_validation_plots()
        logger.info("Thermistor processing complete")

