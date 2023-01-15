# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:05:42 2023

@author: Manuel

Edit missing iNaturalist observation accuracies in batch edit mode.

Requirements: pandas, pyinaturalist, pyperclip (optional, copy to clipboard
automatically)
"""
import pandas as pd
import pyinaturalist
import pkg_resources

# Class iNatEntries to obtain entries w/ missing accuracy for a given user ID
class iNatEntries():
    def __init__(self, uid):
        observations = []
        page_results = [0]
        p = 1
        pp = 30
        print("Gathering observations... this might take a moment.")
        while len(page_results) > 0:
                page = pyinaturalist.get_observations(user_id = uid,
                                                      acc  = False,
                                                      per_page = pp,
                                                      page = p)
                page_results = page["results"]
                observations += page_results
                p += 1
        self.ids = [obs["id"] for obs in observations]

# Function to print out batch edit URLs
def generate_urls(file_path = None, id_obj = None, N = 50):
    installed = [pkg.key for pkg in pkg_resources.working_set]
    if "pyperclip" in installed:
        import pyperclip
        def paste_cb(url):
            pyperclip.copy(url)
            print("Url copied to clipboard.")
    else:
        def paste_cb(url):
            pass
    if file_path is not None:
        df = pd.read_csv(file_path, sep = ";")
        ids = df["id"]
    elif id_obj.__class__.__name__ == "iNatEntries":
        ids = id_obj.ids
    elif type(id_obj).__name__ == "list":
        ids = id_obj
    n_rows = len(ids)
    base_url = "https://www.inaturalist.org/observations/edit/batch?o="
    row = 0
    while row < n_rows:
        end = row + N if row + N < n_rows else n_rows
        iNatIDs = ",".join(str(e) for e in ids[row:end])
        url = base_url
        batch_edit_url = url + iNatIDs
        print(batch_edit_url)
        row += N
        paste_cb(batch_edit_url)
        input("Press Enter to continue...")

# Create instance of the class defined above for your user ID
idobj = iNatEntries(uid = "mrpopp")

# Print out URLs (if module pyperclip available, URLs are copied to clipboard)
generate_urls(id_obj = idobj)
