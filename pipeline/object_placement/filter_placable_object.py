from google import genai
from google.genai import types
from tqdm import tqdm
import json
import os

import re

api_list = [os.environ.get("GEMINI_API_KEY"), 
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GEMINI_API_KEY")]

def decide_object_name_reasonability(object, api_key=api_list[0]):
    
    client = genai.Client(api_key=api_key)
    
    # Read image and audio
    contents = f"Is `{object}` a reasonable object name? Return True if it is, False otherwise. Only return True or False, no other text."
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        # model="gemini-2.5-flash",
        # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
        contents=contents
    )

    return response.text

def decide_if_object_placable(object, affordance_surface='table', api_key=api_list[0]):
    

    client = genai.Client(api_key=api_key)
    
    # Read image and audio
    contents = f"Can I put a/an `{object}` on the `{affordance_surface} given the noraml size and weight? Return True if it is certainly placable, False otherwise or it is uncertain. Only return True or False, no other text."
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        # model="gemini-2.5-flash",
        # model=args.model, # model="gemini-2.0-flash",  # or "gemini-2.5-flash"
        contents=contents
    )

    return response.text

def filter_affordance_surface(object_list):
    object_list_name = object_list.keys()
    extended_object_list = {}
    
    for object in tqdm(object_list_name):
        extended_object_list[object] = {'meshes': object_list[object], 'placable_on_table': decide_if_object_placable(object, 'table'), 'placable_on_chair': decide_if_object_placable(object, 'chair'), 'placable_on_floor': decide_if_object_placable(object, 'floor')}
        print(object, extended_object_list[object]['placable_on_table'], extended_object_list[object]['placable_on_chair'], extended_object_list[object]['placable_on_floor'])
    
    with open('extended_object_list.json', 'w') as f:
        json.dump(extended_object_list, f)

def filter_object_name_and_split():
    with open('extended_object_list.json', 'r') as f:
        extended_object_list = json.load(f)
    
    placable_on_table_list = {}
    placable_on_chair_list = {}
    placable_on_floor_list = {}
    
    for object in tqdm(extended_object_list):
        try:
            name_reasonability = decide_object_name_reasonability(object)
        except:
            print(f"Error in deciding the name reasonability of {object}")
            continue
        
        if name_reasonability == 'True' and extended_object_list[object]['placable_on_table'] == 'True':
            placable_on_table_list[object] = extended_object_list[object]
            
        if name_reasonability == 'True' and extended_object_list[object]['placable_on_chair'] == 'True':
            placable_on_chair_list[object] = extended_object_list[object]
            
        if name_reasonability == 'True' and extended_object_list[object]['placable_on_floor'] == 'True':
            placable_on_floor_list[object] = extended_object_list[object]

    
    os.makedirs('object_placement', exist_ok=True)
    with open('object_placement/placable_on_table_list.json', 'w') as f:
        json.dump(placable_on_table_list, f)
        
    with open('object_placement/placable_on_chair_list.json', 'w') as f:
        json.dump(placable_on_chair_list, f)
        
    with open('object_placement/placable_on_floor_list.json', 'w') as f:
        json.dump(placable_on_floor_list, f)

def append_new_objects(object_list, new_jiawei_scenes="/path/to/tmpdata/", folder_name="jiawei", not_found_objects = {}, new_mapping = {}, save_folder = '/path/to/Taxonomy/Data/SimulationMetadata/objects/objects_list'):
    all_exited_meshes = {}
    
    for object in object_list:

        all_meshes = object_list[object]
        for mesh in all_meshes:
            mesh_name = mesh['object name']
            temp = mesh_name.replace('SM_', '').lower()
            match = re.match(r'^[A-Za-z_]+', temp)
            core_object_name = match.group(0).strip('_') if match else temp
            all_exited_meshes[core_object_name] = object
            # if 'Paint_Spray' in mesh['object name']:
            #     import ipdb; ipdb.set_trace()
    for scene in os.listdir(new_jiawei_scenes):
        if not os.path.isdir(os.path.join(new_jiawei_scenes, scene)) or scene == 'NEW':
            continue
        for instance in os.listdir(os.path.join(new_jiawei_scenes, scene)):
            if not os.path.isdir(os.path.join(new_jiawei_scenes, scene, instance)):
                continue
            object_json_name = "seenable_obj_dict.json"
            with open(os.path.join(new_jiawei_scenes, scene, instance, object_json_name), 'r') as f:
                object_dict = json.load(f)
            for object in object_dict:

                if 'SM_' not in object:
                    continue
                temp = object.replace('SM_', '').lower()
                match = re.match(r'^[A-Za-z_]+', temp)
                core_object_name = match.group(0).strip('_') if match else temp
                if core_object_name not in all_exited_meshes:
                    if core_object_name not in not_found_objects:
                        not_found_objects[core_object_name] = []
                    not_found_objects[core_object_name].append({'scene': scene, 'object name': object})
                else:
                    object_list[all_exited_meshes[core_object_name]].append({'object name': object, 'scene': scene})
                    if all_exited_meshes[core_object_name] not in new_mapping:
                        new_mapping[all_exited_meshes[core_object_name]] = []
                    new_mapping[all_exited_meshes[core_object_name]].append({'object name': object, 'scene': scene})
    # unique
    combined_list = []
    for key in not_found_objects:
        if key in object_list:
            object_list[key].extend(not_found_objects[key])
            if key not in new_mapping:
                new_mapping[key] = []
            new_mapping[key].extend(not_found_objects[key])
            combined_list.append(key)
    for key in combined_list:
        not_found_objects.pop(key)
            
    for key in not_found_objects:
        list_of_dicts = not_found_objects[key]
        list_of_tuples = [(d['scene'], d['object name']) for d in list_of_dicts]
        list_of_tuples = list(set(list_of_tuples))
        not_found_objects[key] = [{'scene': t[0], 'object name': t[1]} for t in list_of_tuples]
        
    for key in object_list:
        list_of_dicts = object_list[key]
        list_of_tuples = [(d['object name'], d['scene']) for d in list_of_dicts]
        list_of_tuples = list(set(list_of_tuples))
        object_list[key] = [{'object name': t[0], 'scene': t[1]} for t in list_of_tuples]
    for key in new_mapping:
        list_of_dicts = new_mapping[key]
        list_of_tuples = [(d['object name'], d['scene']) for d in list_of_dicts]
        list_of_tuples = list(set(list_of_tuples))
        new_mapping[key] = [{'object name': t[0], 'scene': t[1]} for t in list_of_tuples]
    with open(os.path.join(save_folder, 'new_all_object_list.json'), 'w') as f:
        json.dump(object_list, f)
    with open(os.path.join(save_folder, 'not_found_objects.json'), 'w') as f:
        json.dump(not_found_objects, f)
    with open(os.path.join(save_folder, 'new_mapping.json'), 'w') as f:
        json.dump(new_mapping, f)
    return object_list, not_found_objects, new_mapping
        
if __name__ == "__main__":
    #
    current_all_object_list = "/path/to/Taxonomy/Data/SimulationMetadata/objects/objects_list/object_list_v3.json"
    save_folder = '/path/to/Taxonomy/Data/SimulationMetadata/objects/objects_list'
    with open(current_all_object_list, 'r') as f:
        object_list = json.load(f)

    # Step 1: Filter out object that is not placable on the table, chair, or floor
    # filter_affordance_surface(object_list)
    
    # Step 2: Filter out object that is not a reasonable object name
    # filter_object_name_and_split()
    
    # Step 3: append new objects to the list
    not_found_objects, new_mapping = {}, {}
    for folder_name in ['jiawei', 'luoxin', 'zehan', 'additional', 'placement']:
        object_list, not_found_objects, new_mapping = append_new_objects(object_list, f"/path/to/Taxonomy/Data/simulationImage/{folder_name}", folder_name=folder_name, not_found_objects=not_found_objects, new_mapping=new_mapping, save_folder = save_folder)

    