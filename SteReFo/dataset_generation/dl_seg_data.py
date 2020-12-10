#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
#TODO: maybe try do hack the thing do download all in categories  (using random keywords and remove duplicates?)
import json
import urllib
import os
from joblib import Parallel, delayed
import multiprocessing
import click

#%%  get the list of images per category to download, note that 
def _get_urls(api_key, categories, safesearch="True"):
    """
    Get the ulrs of the imaegs to downlaod per category.
    This has a lot of hard params and can change over time, see  https://pixabay.com/api/docs/.
    
    categories: list of categories to be used, choose among "fashion", "nature", 
                "backgrounds", "science", "education", "people", "feelings", 
                "religion", "health", "places", "animals", "industry", "food",
                "computer", "sports", "transportation", "travel", "buildings", 
                "business", "music". 
    api_key: you api access key, apply on https://pixabay.com
    safesearch: remove butts from the downloaded images
    """

    #results per page, set to max (200)
    per_page=200 #max
    #final list of urls per categry
    urls = {}
    for c in categories:
        page = 1
        do_request=True
        urls_category = []
        while(do_request):
            print("Building request for category %s page %d"%(c, page))
            #make the request see https://pixabay.com/api/docs/#
            request = "https://pixabay.com/api/?key="+api_key+"&colors=transparent&image_type=photo&category="+c+"&per_page="+str(per_page)+"&page="+str(page)+"&safesearch="+safesearch
            print(request)
            #do the reqest
            json_return = urllib.request.urlopen(request).read()
            #parse the result
            parsed_json = json.loads(json_return)
            nb_return = len(parsed_json["hits"])
            assert(nb_return!=0)
            print("Parsing json %s page %d"%(c, page))
            max_nb_imgs = parsed_json["totalHits"]#note that is limiting us to 500 images per category
            if page*per_page>max_nb_imgs:
                do_request=False
            page+=1
            #extract the urls
            urls_category+=[parsed_json["hits"][i]["largeImageURL"] for i in range(nb_return) ]
        urls[c]=urls_category

#%% download the images to a given folder
def _do_download(urls, categories, out_folder,  num_cores = None):
    """
    Download the images from the urls.
    Uses one thread per category. By default uses all the availables cpu cores.
    """
    def dowload_category(c):
        print("Downloading %d images for category %s"%(len(urls[c]), c))
        os.makedirs(out_folder+"/"+c)
        for i in range(len(urls[c])):
            print("Dowloading image %05d"%i)
            urllib.request.urlretrieve(urls[c][i], out_folder+"/"+c+"/%05d.jpg"%i)
    if num_cores is None:           
        num_cores = multiprocessing.cpu_count()  
    _ = Parallel(n_jobs=num_cores)(delayed(dowload_category)(i) for i in categories)
    


#%%Hard parameters, can change with the api, check https://pixabay.com/api/docs/
@click.command()

@click.argument('api_key')#mibe is 11645072-19ccb825b84147eddb0a89a12
@click.option('--categories', '-c', multiple=True, help='Category to download, dont use it to download all or choose among "fashion", "nature", "backgrounds", "science", "education", "people", "feelings", "religion", "health", "places", "animals", "industry", "food", "computer", "sports", "transportation", "travel", "buildings", "business", "music". ')
@click.option('--out_folder', '-o', default=".", help='Output folder to dowload the dataset into')
@click.option('--safesearch', '-s', is_flag=True, help='Use this to remove butts from the dowloaded images.')

def prepare_dataset(api_key, categories, out_folder, safesearch):
    print("Hello :)")
    print(categories)
    if categories == ():
        categories =  ["fashion", "nature", "backgrounds", "science", "education", 
                   "people", "feelings", "religion", "health", "places", "animals",
                   "industry", "food", "computer", "sports", "transportation", 
                   "travel", "buildings", "business", "music"]
    if safesearch:
        safesearch="True"
    else:
        safesearch="False"
    print("Getting download urls")
    urls=_get_urls(api_key, categories, safesearch)
    print("Downloading")
    _do_download(urls, categories, out_folder)
    print("Bye :)")

if __name__ == '__main__':
    prepare_dataset()

