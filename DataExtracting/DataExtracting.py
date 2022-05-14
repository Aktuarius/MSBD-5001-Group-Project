import ee
import json
from pathlib import Path
import httplib2
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as disp
from osgeo import gdal
import time
from datetime import datetime

class ProcessAndExport:
    def __init__(self,collection_Name,bands_idx):
        self.collection_name = collection_Name
        self.bands_idx = bands_idx
        
    def appendBands(self,now,running):
        running = ee.Image(running)
        now = now.select(self.bands_idx)
        accum = ee.Algorithms.If(ee.Algorithms.IsEqual(running,None),now,running.addBands(ee.Image(now)))
        return accum

    def exportImage(img,desc,region,type_name):
        task = ee.batch.Export.image.toDrive(image=img,folder = f"{desc}_{type_name}",description = desc,region=region,scale=500,crs = 'EPSG:4326')
        task.start()
        while task.status()['state'] == "RUNNING":
            print("Running...")
            time.sleep(10)
        print(f"Done,{task.status()}")


class ObtainGEEdata:
    def __init__(self,collection_name,date1,date2,countries,geojson_path,bands_idx,type_name):
        if type(geojson_path) == str and Path(geojson_path).exists():
            self.geojson_path = geojson_path
            self.collection_name = collection_name
            if datetime.strptime(date1,"%Y-%m-%d") < datetime.strptime(date2,"%Y-%m-%d"):
                self.date1 = date1
                self.date2 = date2
            else:
                raise Exception("Date1 cannot be after Date2")
            self.countries = countries
            self.bands_idx = bands_idx
            self.type_name = type_name
            self.get_coordinates()
            self.extract_collections()
        else:
            raise FileNotFoundError("GeoJson file does not exist")

    def get_coordinates(self):
        with open(self.geojson_path) as f:
            data = json.load(f)
        self.coords = {}
        for j in self.countries:
            for i in data['features']:
                if i['properties']['name'] == j:
                    self.coords[j] = i['geometry']['coordinates']
                    break

    def extract_collections(self):
        for country in self.countries:
            aoi = ee.Geometry.MultiPolygon(self.coords[country])
            imgcoll = ee.ImageCollection(self.collection_name)\
                                        .filterBounds(aoi)\
                                        .filterDate(ee.Date(self.date1),ee.Date(self.date2))
            print(f"Number of images : {imgcoll.size().getInfo()}")
            processAndexport = ProcessAndExport(collection_Name=self.collection_name,bands_idx=self.bands_idx)
            img = imgcoll.iterate(processAndexport.appendBands)
            img = ee.Image(img)
            try:
                processAndexport.exportImage(img,country,aoi,self.type_name)
            except:
                print(f'Failed:{country}')


if __name__ == "__main__":
    #Google Earth Engine Authentication
    ee.Authenticate()
    _http_transport = httplib2.Http(disable_ssl_certificate_validation=True)
    ee.Initialize(http_transport=_http_transport)

    geojson_path = list(Path(Path.cwd()/"Data"/"GeoJson").resolve().iterdir())[0]
    countries = ["South Africa","Ethiopia","Malawi"]
    surface_extract = ObtainGEEdata(collection_name='MODIS/MOD09A1',date1='2002-01-01',date2='2016-01-01',countries=countries,bands_idx=[0,1,2,3,4,5,6],type_name = "Surface")
    landcover_extract = ObtainGEEdata(collection_name='MODIS/006/MCD12Q1',date1='2002-01-01',date2='2016-01-01',countries=countries,bands_idx=[0],type_name = "LandCover")
    temp_extract = ObtainGEEdata(collection_name='MODIS/MYD11A2',date1='2002-07-04',date2='2016-07-03',countries=countries,bands_idx=[0,4],type_name = "Temperature")    
    water_extract = ObtainGEEdata(collection_name='MODIS/006/MOD44W',date1='2002-01-01',date2='2016-01-01',countries=countries,bands_idx=[0,1],type_name = "WaterCover")
    