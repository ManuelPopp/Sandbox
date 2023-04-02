#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:10:48 2023
"""
__author__ = "Manuel"
__date__ = "Sun Apr  2 15:10:48 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Production"

#-----------------------------------------------------------------------------|
# Import modules
import os, requests
import pandas as pd
import threading as th
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

#-----------------------------------------------------------------------------|
# Functions

def download(remote_url, local_file):
    data = requests.get(remote_url)
    try:
        with open(local_file, "wb") as f:
            f.write(data.content)
        exit_status = 0
    except:
        exit_status = 1
    return exit_status

#-----------------------------------------------------------------------------|
# GUI

class BulkDownloader(tk.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.pack(padx = 0, pady = 0)
        self.create_widgets()
        self.master.minsize(300, 50)
        self.master.iconphoto(False, tk.PhotoImage(file = "icons/main.png"))
        self._current_progress = 0
        self._downloading = False
        self._t2 = th.Thread(target = self.download_files, daemon = True)
    
    def create_widgets(self):
        # Variables
        ## Input file
        self.file_name = tk.StringVar()
        self.file_name.set("")
        
        ## Output folder
        self.output_folder = tk.StringVar()
        self.output_folder.set("")
        
        ## Progress bar status
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)
        
        # Title
        self.winfo_toplevel().title("Bulk Downloader")
        
        # Display
        self.display_current_in = tk.Entry(self)
        self.display_current_in.grid(row = 0, column = 4, columnspan = 4,
                                  sticky = "EW")
        self.display_current_in["textvariable"] = self.file_name
        
        self.display_current_out = tk.Entry(self)
        self.display_current_out.grid(row = 1, column = 4, columnspan = 4,
                                  sticky = "EW")
        self.display_current_out["textvariable"] = self.output_folder
        
        # Buttons
        ## Select input file
        self.select_in_button = tk.Button(self)
        self.select_in_button["text"] = "Select input file"
        self.select_in_button["command"] = self.select_in_file
        self.select_in_button.grid(row = 0, column = 0, columnspan = 3,
                                   sticky = "EW")
        
        ## Select output directory
        self.select_out_button = tk.Button(self)
        self.select_out_button["text"] = "Select output directory"
        self.select_out_button["command"] = self.select_out_folder
        self.select_out_button.grid(row = 1, column = 0, columnspan = 3,
                                 sticky = "EW")
        
        ## Close main window
        self.cancel_button = tk.Button(self)
        self.cancel_button["text"] = "Cancel"
        self.cancel_button["command"] = self.cancel_all#.master.destroy
        self.cancel_button.grid(row = 2, column = 0, columnspan = 3,
                                sticky = "EW")
        
        ## Download files
        self.download_button = tk.Button(self)
        self.download_button["text"] = "Download"
        self.download_button["command"] = self.start_download
        self.download_button.grid(row = 2, column = 4, columnspan = 4,
                                  sticky = "E")
        
        ## Progress bar
        self.progress = ttk.Progressbar(self, variable = self.progress_var,
                                        mode = "determinate")
        self.progress.grid(row = 3, column = 0, columnspan = 38, sticky = "EW")
    
    def select_in_file(self):
        current_selection = self.file_name.get()
        set_dir_to = current_selection if current_selection != "" else "/"
        file_types = (("csv file", "*.csv"), ("all files", "*.*"))
        selection = fd.askopenfilename(title = "Open file",
                                       initialdir = set_dir_to,
                                       filetypes = file_types)
        self.file_name.set(selection)
    
    def select_out_folder(self):
        current_selection = self.output_folder.get()
        set_dir_to = current_selection if current_selection != "" else "/"
        selection = fd.askdirectory(title = "Select destination directory",
                                       initialdir = set_dir_to)
        self.output_folder.set(selection)
    
    def download_files(self):
        local_folder = self.output_folder.get()
        csv_table = self.file_name.get()
        
        table = pd.read_csv(csv_table)
        url_list = table[table.columns[0]].to_list()
        
        for enumerator, item in enumerate(url_list):
            if not self._continue_loop:
                break
            local_file = os.path.join(local_folder, os.path.split(item)[1])
            exit_status = download(item, local_file)
            
            self._current_progress = ((enumerator + 1) * 100) // len(url_list)
            self.progress_var.set(self._current_progress)
        self.finished(exit_status)
    
    def start_download(self):
        self._continue_loop = True
        if not self._downloading:
            self._current_progress = 0
            self._t2.start()
    
    def finished(self, exit_status):
        self._downloading = False
        
        messageWindow = tk.Toplevel(self)
        window_title = "Bulk Downloader finished" if exit_status == 0 else \
            "Bulk download failed"
        messageWindow.title(window_title)
        tk.Label(messageWindow, text = window_title).pack()
        tk.Button(messageWindow, text = "Close window",
                  command = self.master.destroy).pack()
    
    def cancel_all(self):
        if self._downloading:
            self._t2.terminate()
        self._downloading = False
        self._current_progress = 0
        self.master.destroy()

#-----------------------------------------------------------------------------|
# Run Bulk Downloader
root = tk.Tk()
app = BulkDownloader(root)
app.mainloop()