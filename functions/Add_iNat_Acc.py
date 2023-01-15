# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:05:42 2023

@author: Manuel

Edit missing iNaturalist observation accuracies in batch edit mode.
- Replace "mrpopp" with your username in line 76.
- Run the script to view the first batch edit URLs.
- Make sure you are logged-in to iNaturalist and enter the URL in your browser.
- On the website:
    - Expand "Batch Operations"
    - Enter a reasonable value in the "Acc (m)" field
    - Click on "Apply"
    - At the bottom of the page, click on "Save all"
- Come back and hit ENTER to generate the next URL.
- Repeat until the script has finished.

Requirements: pandas, pyinaturalist, pyperclip (optional, copy to clipboard
automatically)
"""
import pandas as pd
import pyinaturalist
import pkg_resources

# Class iNatEntries to obtain entries w/ missing accuracy for a given user ID
class iNatEntries():
    def __init__(self, uid):
        self.uid = uid
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
    
    # Re-initialise to update the observation ID list
    def refresh(self):
        self.__init__(uid = self.uid)
    
    # Generate URLs
    def generate_urls(self, N = 50):
        installed = [pkg.key for pkg in pkg_resources.working_set]
        if "pyperclip" in installed:
            import pyperclip
            def paste_cb(url):
                pyperclip.copy(url)
                print("\n(URL copied to clipboard.)")
        else:
            def paste_cb(url):
                pass
        
        ids = self.ids
        n_rows = len(ids)
        base_url = "https://www.inaturalist.org/observations/edit/batch?o="
        row = 0
        print("Hint: Cancel and adjust parameter 'N' to edit more " + \
              "observations at a time. Try a smaller value in case " + \
                  "of issues concerning the batch update.")
        while row < n_rows:
            end = row + N if row + N < n_rows else n_rows
            iNatIDs = ",".join(str(e) for e in ids[row:end])
            url = base_url
            batch_edit_url = url + iNatIDs
            mssg0 = "\n\nOutput for observations {0} to {1} (total: {2}). " + \
                "Visit the following URL to batch-edit:\n\n{3}"
            print(mssg0.format((row + 1), end, n_rows, batch_edit_url))
            row += N
            paste_cb(batch_edit_url)
            if row < n_rows:
                mssg1 = "Press ENTER to continue or enter 'c' to cancel. "
                user_in = input(mssg1)
                if user_in.lower() == "c":
                    print("Process cancelled.")
                    break
            else:
                print("Finished.")

# Create instance of the class defined above for your user ID
idobj = iNatEntries(uid = "mrpopp")

# Print out URLs (if module pyperclip available, URLs are copied to clipboard)
idobj.generate_urls(N = 15)

'''
# Function to print out batch edit URLs from either an iNatEntries instance or
# an exported .csv file containing observation IDs.
# This is an alternative to the .generate_urls() method of the above class.
# It also accepts a iNaturalist-exported .csv file or a list containing obser-
# vation IDs
def generate_urls(id_obj = None, file_path = None, N = 50):
    installed = [pkg.key for pkg in pkg_resources.working_set]
    if "pyperclip" in installed:
        import pyperclip
        def paste_cb(url):
            pyperclip.copy(url)
            print("\n(Url copied to clipboard.)")
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
        print("Visit the following URL:\n\n" + batch_edit_url)
        row += N
        paste_cb(batch_edit_url)
        input("Press Enter to continue...")
    print("Finished.")
'''
