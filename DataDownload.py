import argparse
import os
from netrc import netrc
from subprocess import Popen
from getpass import getpass 
import requests
from pathlib import Path

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description = "Download MODIS Data")
    # parser.add_argument("-dir","--directory",required = True)
    # args = parser.parse_args()

    savepath = os.path.join('D:\GitHub\MSBD-5001-Group-Project\DataExtracting',"Data")
    files = os.listdir(savepath)
    print('Save path: ', savepath)
    urs = 'urs.earthdata.nasa.gov'    # Address to call for authentication
    prompts = ['Enter NASA Earthdata Login Username: ',
               'Enter NASA Earthdata Login Password: ']
    # --------------------------------AUTHENTICATION CONFIGURATION----------------------------------- #
    # Determine if netrc file exists, and if so, if it includes NASA Earthdata Login Credentials


    # --------------------------------READ FILES TO DOWNLOAD----------------------------------------- #
    #& files[0].endswi
    # th('.txt')
    if len(files) == 2:
        filelist = open(os.path.join(args.directory,"Data",files[1]),'r').readlines()
        print(filelist)
        homeDir = os.path.join(os.path.join(args.directory,"Data"))
        if savepath[-1] != "\\":
            savepath = savepath + os.sep
            print(savepath)
        os.chdir(homeDir)


        if Path(os.path.join(homeDir,".netrc")).exists() == False:
        # Popen('cd {}'.format(homeDir))
            Popen('type nul > {0} | echo machine {1} >> {0}'.format(".netrc", urs), shell=True)
            Popen('echo login {} >> {}'.format(getpass(prompt=prompts[0]), ".netrc"), shell=True)
            Popen('echo password {} >> {}'.format(getpass(prompt=prompts[1]),".netrc"), shell=True)

        netrcpath = os.path.join(args.directory,"Data",".netrc")
        print(netrcpath)
        for file in filelist:
            saveName = os.path.join(savepath,file.split('/')[-1].strip())
            print(file.strip())
            #output of netrc(netrcpath).authenticators(hostname) is ('Username', None, 'Password')
            with requests.get(file.strip(),verify = False,stream = True,auth = (netrc(netrcpath).authenticators('urs.earthdata.nasa.gov')[0],netrc(netrcpath).authenticators('urs.earthdata.nasa.gov')[2])) as response:
                if response.status_code != 200:
                    with open(os.path.join(args.directory,"log_fail.txt"),"a") as txt_file:
                        txt_file.write("{} not downloaded".format(file.split('/')[-1].strip()))
                else:
                    response.raw.decode_content = True
                    content = response.raw
                    with open(saveName, 'wb') as d:
                        while True:
                            chunk = content.read(16 * 1024)
                            if not chunk:
                                break
                            d.write(chunk)
                    with open(os.path.join(args.directory,"log_works.txt"),"a") as txt_file:
                            txt_file.write("{} downloaded".format(file.split('/')[-1].strip()))
    else:
        print("Download Directory should only contain one file and it should be txt file")























    # """
    # ---------------------------------------------------------------------------------------------------
    #  How to Access the LP DAAC Data Pool with Python
    #  The following Python code example demonstrates how to configure a connection to download data from
    #  an Earthdata Login enabled server, specifically the LP DAAC's Data Pool.
    # ---------------------------------------------------------------------------------------------------
    #  Author: Cole Krehbiel
    #  Last Updated: 05/14/2020
    # ---------------------------------------------------------------------------------------------------
    # """
# # Load necessary packages into Python
# from subprocess import Popen
# from getpass import getpass
# from netrc import netrc
# import argparse
# import time
# import os
# import requests

# # ----------------------------------USER-DEFINED VARIABLES--------------------------------------- #
# # Set up command line arguments
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-dir', '--directory', required=True, help='Specify directory to save files to')
# parser.add_argument('-f', '--files', required=True, help='A single granule URL, or the location of csv or textfile containing granule URLs')
# args = parser.parse_args()

# saveDir = args.directory  # Set local directory to download to
# files = args.files        # Define file(s) to download from the LP DAAC Data Pool
# prompts = ['Enter NASA Earthdata Login Username \n(or create an account at urs.earthdata.nasa.gov): ',
#            'Enter NASA Earthdata Login Password: ']

# # ---------------------------------SET UP WORKSPACE---------------------------------------------- #
# # Create a list of files to download based on input type of files above
# if files.endswith('.txt') or files.endswith('.csv'):
#     fileList = open(files, 'r').readlines()  # If input is textfile w file URLs
# elif isinstance(files, str):
#     fileList = [files]                       # If input is a single file

# # Generalize download directory
# if saveDir[-1] != '/' and saveDir[-1] != '\\':
#     saveDir = saveDir.strip("'").strip('"') + os.sep
# urs = 'urs.earthdata.nasa.gov'    # Address to call for authentication

# # --------------------------------AUTHENTICATION CONFIGURATION----------------------------------- #
# # Determine if netrc file exists, and if so, if it includes NASA Earthdata Login Credentials
# try:
#     netrcDir = os.path.expanduser("~\.netrc")
#     netrc(netrcDir).authenticators(urs)[0]

# # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
# except FileNotFoundError:
#     homeDir = os.path.expanduser("~")
#     Popen('type nul > {0}.netrc | echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
#     Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
#     Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)

# # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
# except TypeError:
#     homeDir = os.path.expanduser("~")
#     Popen('echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
#     Popen('echo login {} >> {}.netrc'.format(getpass(prompt=prompts[0]), homeDir + os.sep), shell=True)
#     Popen('echo password {} >> {}.netrc'.format(getpass(prompt=prompts[1]), homeDir + os.sep), shell=True)

# # Delay for up to 1 minute to allow user to submit username and password before continuing
# # tries = 0
# # while tries < 30:
# #     try:
# #         netrc(netrcDir).authenticators(urs)[2]
# #     except:
# #         time.sleep(2.0)
# #     tries += 1

# # # -----------------------------------------DOWNLOAD FILE(S)-------------------------------------- #
# # # Loop through and download all files to the directory specified above, and keeping same filenames
# # for f in fileList:
# #     if not os.path.exists(saveDir):
# #         os.makedirs(saveDir)
# #     saveName = os.path.join(saveDir, f.split('/')[-1].strip())

# #     # Create and submit request and download file
# #     with requests.get(f.strip(), verify=False, stream=True, auth=(netrc(netrcDir).authenticators(urs)[0], netrc(netrcDir).authenticators(urs)[2])) as response:
# #         if response.status_code != 200:
# #             print("{} not downloaded. Verify that your username and password are correct in {}".format(f.split('/')[-1].strip(), netrcDir))
# #         else:
# #             response.raw.decode_content = True
# #             content = response.raw
# #             with open(saveName, 'wb') as d:
# #                 while True:
# #                     chunk = content.read(16 * 1024)
# #                     if not chunk:
# #                         break
# #                     d.write(chunk)
# #             print('Downloaded file: {}'.format(saveName))
# #     #time.sleep(1.0)