#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import json
import os 
from tinydb import TinyDB, Query
 
current_dir = Path(os.getcwd()).absolute()
results_dir = current_dir.joinpath('results')
kv_data_dir = results_dir.joinpath('kvdb')
kv_data_dir.mkdir(parents=True, exist_ok=True)

def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)
    
class DocumentDB(object):
    def __init__(self, db_path):
        ## You can use the code from the previous exmaple if you would like
        people_json = kv_data_dir.joinpath('people.json')  
        visited_json = kv_data_dir.joinpath('visited.json')
        sites_json = kv_data_dir.joinpath('sites.json')
        measurements_json = kv_data_dir.joinpath('measurements.json') 
        
        self._db_path = Path(db_path)
        self._db = None
        
        self.people_json = load_json(people_json)
        self.visited_json = load_json(visited_json)
        self.sites_json = load_json(sites_json)
        self.measurements_json = load_json(measurements_json)
        self._load_db()
    
    def Merge(dict1, dict2): 
        res = {**dict1, **dict2}
        return res
    
    def _load_db(self):
        self._db = TinyDB(self._db_path) 
        data_table = self._db.table('PatientRecord')

        Q1 = Query() 
        visits = []
        measurements = []
        people = []
        sites = []
        #merged_data = [] 
        
        for people_key, people_values in self.people_json.items(): 
            peoples_dict = {}
            merged_dict = {}

            merged_dict.update(people_values) 

            for measurement_key, measurement_value in self.measurements_json.items(): 
                measurements_dict = {}
                for visit_key, visit_value in self.visited_json.items():  
                    visits_dict = {}
                    if people_values['person_id'] == measurement_value['person_id']: 
                        measurements_dict = measurement_value
                        if measurement_value['visit_id'] == visit_value['visit_id']: 
                            visits_dict = visit_value
                            for site_key, site_value in self.sites_json.items():
                                sites_dict = {}
                                if visit_value['site_id'] == site_value['site_id']:
                                    sites = site_value 
                                    visit_values_2 = {}
                                    visit_values_2 = visits_dict 

                                    if len(sites)>0:
                                        visit_values_2.update({"site": sites})
                                    if len(measurements_dict) > 0:
                                        visit_values_2.update({"measurements" : [measurements_dict]}) 
                                    merged_dict.update({"visits" : visit_values_2})  
                                    data_table.insert(merged_dict)  
        # Test Queries
        print('SEARCH A ROW IN TABLE \n',data_table.search(Q1.person_id == 'lake'))
        for item in data_table:
                print('\n\n PRINT EACH ROW IN TABLE \n',item)
        print('\n\n PRINT ENTIRE TABLE \n\n',data_table.all())


# In[2]:


db_path = results_dir.joinpath('patient-info.json') 
if db_path.exists():
    os.remove(db_path)

db = DocumentDB(db_path)

