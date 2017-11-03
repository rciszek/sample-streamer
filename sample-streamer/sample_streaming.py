"""
@author: rciszek
"""

import numpy as np
from multiprocessing import Lock,Process,Event
import h5py
import subprocess
import os
import time
import logging


def list_buffer_files(buffer_folder_path, buffer_name):
    """ 
    Lists all simulation data files in the buffer folder. Provides a list of
    tuples which contain the file name and the unix timestamp of the last 
    modification time. The tuples are sorted in ascending order according to
    the modification time.
    
    Arguments:
        buffer_folder_path: The path to the buffer folder
        buffer_name: Task specific buffer name 
    """  
          
    #Find all the x and y files in the buffer folder
    x_paths = [os.path.join(buffer_folder_path,file_name) for file_name in os.listdir(buffer_folder_path) if file_name.endswith(buffer_name + "_x.hdf5")]    
    y_paths = [os.path.join(buffer_folder_path,file_name) for file_name in os.listdir(buffer_folder_path) if file_name.endswith(buffer_name + "_y.hdf5")] 
    
    #Get the modification times for all files
    x_modification_times = [ os.path.getmtime(file_path) for file_path in x_paths ] 
    y_modification_times = [ os.path.getmtime(file_path) for file_path in y_paths ] 
    
    #Sort the files by timestamps
    x_file_data = sorted( zip(x_paths,x_modification_times),key=lambda k:k[1])           
    y_file_data = sorted( zip(y_paths,y_modification_times),key=lambda k:k[1])   
 
    if len(x_paths) == 0 or len(y_paths) == 0:
        return []
    else:
     return x_file_data,y_file_data
 
   
def calculate_buffer_size(buffer_folder_path, buffer_name):
    """
    Calculates the current size of the buffer, i.e. the number of files matching the buffer name
    
    Arguments:
        buffer_folder_path: The path to the buffer folder
        buffer_name: Task specific buffer name 
    
    """ 
    x_paths = [os.path.join(buffer_folder_path,file_name) for file_name in os.listdir(buffer_folder_path) if file_name.endswith(buffer_name + "_x.hdf5")]    
    y_paths = [os.path.join(buffer_folder_path,file_name) for file_name in os.listdir(buffer_folder_path) if file_name.endswith(buffer_name + "_y.hdf5")] 
    
    #Return the minimumf the amount of x and y file to avoid possible cases where
    #x is written but y has not yet been writen.
    return np.min([ len(x_paths), len(y_paths)])
    
       
def store_hashes(data_buffer, hash_lock, hash_storage_file):
    """
    Calculates sample hashes and stores them thread safely to the target hdf5 file
    """ 
    logging.getLogger('sample_generator.store_hashes').debug("Calculating hashes")      
    hashes = np.zeros((data_buffer.shape[0],1));
    for i in range(0,data_buffer.shape[0]):
        hashes[i,0] = hash(data_buffer[i].data.tobytes())
    
    hash_lock.acquire()
    try:
        logging.getLogger('sample_generator.store_hashes').debug("Writing hashes")  
        with open(hash_storage_file, "a") as hash_storage: 
                hash_storage.writelines([ str("%1.0f"% value) + "\n" for value in hashes[:,0] ])    
    finally:
        hash_lock.release()
    logging.getLogger('sample_generator.store_hashes').debug("Hash storage completed")     
  
class GeneratorThread(Process):
    """
    A thread which calls Octave when necessary in order to generate data set files to buffer folder.
    """      
    
    def __init__(self, group=None, target=None, name=None,args=(), kwargs=None, verbose=None ):
        super().__init__()   
        self.args = args
        self.kwargs = kwargs   
        
    def run(self):
        
        self.generate_buffer_file(*self.args)
    
    def generate_buffer_file(self,buffer_size, buffer_files, buffer_path,m_file_path,disk_check_interval, buffer_name, end_signal ):             
    
        while end_signal.is_set() == False:
            #Execute the given Octave/MATLAB scriot using Octave if the number
            #of dataset files in the buffer folder is below the threshold
            if calculate_buffer_size(buffer_path,buffer_name) < buffer_files:  
                logging.getLogger('sample_generator.generator_thread').debug("Buffer file number below threshold. Generating buffer files.")     
                   
                #Execute .m file using Octave.
                process = subprocess.Popen(["octave", m_file_path, buffer_path, str(buffer_size),buffer_name],stdout=subprocess.PIPE)
                process.wait()
                logging.getLogger('sample_generator.generator_thread').debug("Buffer file generation completed with: %s"%(process.stdout.read()))                          
                 
            time.sleep(disk_check_interval)  
            
        logging.getLogger('sample_generator').debug("Process %s stopped"%(self.name))   
        

class SampleGenerator:
    """
    A generator class which encapsulates the management of disk and memory buffers.
    
        Arguments:
            batch_size: The number of samples to be returned on each request
            buffer_size: The total number samples in the buffer
            buffer_files: The minimum number of buffer files to be kept on the  
                buffer folder.
            n_processes: Number of generator processes used to generate buffer files.
            buffer_path: Path of the buffer file folder.
            m_file_path: Path of the .m file used to generate the buffer files. Note
                that the .m file has to store the generated files to the same buffer 
                folder defined in buffer_path 
           disk_check_interval: The interval in seconds of thechecks performed to 
               ensure the sufficient amount of buffer files.
           buffer_name: Task specific name for the buffer. Equally named tasks will
               use the same buffer files.
           verbosity: Logging verbosity. Zero or less for no output, 1 or more for
               logging to both file and console.
    """  
    
    def __init__(self, batch_size=32, 
                 buffer_size=1000, 
                 buffer_files=4,n_processes=1, 
                 buffer_path="buffer/", 
                 m_file_path="", 
                 disk_check_interval = 10.0, 
                 buffer_name="default",
                 verbosity=0):

        self.__setup_logger(verbosity)
        logging.getLogger('sample_generator').debug("Initializing")      
        self.__setup_generator_processes(n_processes,buffer_size,buffer_files,buffer_path,m_file_path,disk_check_interval,buffer_name)
    
        self.buffer = None
        self.back_buffer = None
        self.current_index = 0
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_path = buffer_path
        self.m_file_path = m_file_path    
        self.disk_check_interval = disk_check_interval
        self.buffer_name = buffer_name
        self.hash_lock = Lock() 
        self.initialize_buffer()


  
    def __setup_generator_processes(self, n_processes,buffer_size,buffer_files,buffer_path,m_file_path,disk_check_interval,buffer_name):  
        """ 
        Creates and starts generator processes
        
        Arguments:
            n_processes: Number of generator processes used to generate buffer files.            
            buffer_size: The total number samples in the buffer
            buffer_files: The minimum number of buffer files to be kept on the  
                buffer folder.
            buffer_path: Path of the buffer file folder.
            m_file_path: Path of the .m file used to generate the buffer files. Note
                that the .m file has to store the generated files to the same buffer 
                folder defined in buffer_path        
           disk_check_interval: The interval in seconds of thechecks performed to 
               ensure the sufficient amount of buffer files.                
        """
        
        self.end_signal = Event()
        
        for i in range(0,n_processes):
            generator_thread = GeneratorThread( args=(buffer_size,buffer_files,buffer_path,m_file_path,disk_check_interval,buffer_name,self.end_signal), name='buffer_file_generator_' + str(i))  
            generator_thread.start()       
        
  
    def __setup_logger(self, verbosity):
        
        stream_level = logging.ERROR
        file_level = logging.ERROR
        
        if verbosity == 1:
            stream_level = logging.DEBUG
            file_level = logging.DEBUG            
        
        logger = logging.getLogger('sample_generator')        
        logger.setLevel(logging.DEBUG)     
        
        file_handler = logging.FileHandler('generator.log')
        file_handler.setLevel(file_level)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_level)
        
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)    
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if self.current_index >= self.buffer_size:
            logging.getLogger('sample_generator').debug("Buffer exhausted, swapping")            
            self.swap_buffers()
            self.current_index = 0
        batch = self.buffer[0][self.current_index:self.current_index+self.batch_size],self.buffer[1][self.current_index:self.current_index+self.batch_size]
        self.current_index += self.batch_size
        logging.getLogger('sample_generator').debug("Returned batch %i"%(self.current_index))              
        return batch
       
    def initialize_buffer(self):
        """
        Fills both the back and front buffers.
        """
        logging.getLogger('sample_generator').debug("Initializing buffer")         
        #Load file to back buffer
        self.update_back_buffer()
        #Swao buffers
        self.swap_buffers()
        #Now both front and back buffer are full
        logging.getLogger('sample_generator').debug("Buffer initialized")            
         
    def swap_buffers(self):
        """
        Fills the front buffer with the data from the back buffer and refills the
        back buffer with new data
        """
        logging.getLogger('sample_generator').debug("Swapping buffers")                
        self.buffer = self.back_buffer
        self.update_back_buffer()

        
        
    def update_back_buffer(self):     
        """
        Fills the back buffer by reading data from the buffer folder
        """
        #If the buffer file folder is empty, wait for the folder to be updated
        if calculate_buffer_size(self.buffer_path,self.buffer_name) == 0:           
            while True:
                logging.getLogger('sample_generator').debug("NO BUFFER FILES AVAILABLE. Waiting for buffer files")                    
                time.sleep(self.disk_check_interval)
                if len(list_buffer_files(self.buffer_path, self.buffer_name)) > 0:                   
                    break
    
  
        self.back_buffer = self.read_buffer_file()
        
        hash_process = Process(target=store_hashes, args=(self.back_buffer[0], self.hash_lock, self.buffer_name + "_hashes.txt"))
        hash_process.start()
        
    def read_buffer_file(self):
        """
        Reads data from buffer files and deletes the read files
        """
        logging.getLogger('sample_generator').debug("Loading buffer")     
        #List the available buffer files
        x_file_data, y_file_data = list_buffer_files(self.buffer_path, self.buffer_name)        

        #Load the oldest buffer data      
        hdf5 = h5py.File(x_file_data[0][0], 'r')
        x = hdf5['phi_final_mem']['value'][:]
        hdf5.close()
        x = np.expand_dims(x,axis=5) 
        hdf5 = h5py.File(y_file_data[0][0], 'r')        
        y = hdf5['chi_target_mem']['value'][:]
        hdf5.close() 
        y = np.expand_dims(y,axis=5)  
        self.back_buffer = [x,y]
        
        #Remove the read files
        os.remove(x_file_data[0][0])
        os.remove(y_file_data[0][0])
        
        return [x,y]
    
    def stop(self):
        """
        Stops the generator processses
        """
        logging.getLogger('sample_generator').debug("Sending stop signal to generator processes")             
        self.end_signal.set()
        
        