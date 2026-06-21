#!/usr/bin/env python3
"""
Filter Utilities for QA Generation

Contains filtering functions to exclude certain object types from QA generation.
"""

from typing import Dict, List, Any



# Clusters to exclude from QA generation for specific question types
# These are the EXACT cluster names from the taxonomy files
VOID_CLUSTERS = {
    'material': [
        'Abstract / Depictions / Scenes/ Occupations',  # Abstract/depicted objects, not real materials
        'Unclassified',  # Objects without material classification
    ],
    'function': [
        'No Clear Function',  # Void function cluster
    ],
    'physical': [
        'No-Physical-Properties',  # Objects without physical properties
    ],
    'affordance': [
        'Human Roles & Identities (Occupations/Person Types)',
        'Natural Scenes (View/Appraise)',
        'Phenomena (View/Read/Appraise)',
    ],
    'texture': [
        'Atextural/Symbolic (Documents/Icons/Light-Only Events)',
        'Unclassified',
    ],
    'shape': [
        'Organic / Abstract / No Definite Shape (Animals/Humans/Roles/Concepts)',
    ]
}

# High-reliability object classes for geometric orientation-based spatial questions
# These classes have semantically meaningful front/back/left/right orientation
# All objects in "Vehicles, Transportation" and "Roles, Occupation, and Directed Actions" 
# taxonomy categories are included, plus additional objects with clear orientation
HIGH_RELIABILITY_CLASSES = {
    # Vehicles, Transportation - All vehicles have clear front/back orientation
    'aircraft carrier', 'airliner', 'army tank', 'atv', 'baby carriage', 'barge',
    'battleship', 'bicycle', 'biplane', 'boat', 'bus', 'cable car', 'camper',
    'canoe', 'car', 'cargo ship', 'cart', 'catamaran', 'city bus', 'concept car',
    'convertible', 'coupe', 'cruise ship', 'decker bus', 'dirt bike',
    'emergency vehicle', 'escalator', 'excavator', 'ferry', 'fishing boat',
    'food truck', 'forklift', 'garbage truck', 'glider', 'helicopter',
    'horse cart', 'horseback', 'hot air balloon', 'jeep', 'jet', 'kayak',
    'limo', 'luxury yacht', 'minivan', 'mobility scooter', 'monster truck',
    'moped', 'motorbike', 'motorboat', 'motorcycle', 'muscle car',
    'passenger train', 'plane', 'police car', 'push cart', 'race car', 'raft',
    'recumbent', 'rickshaw', 'river boat', 'rowboat', 'sailboat', 'scooter',
    'seaplane', 'sedan', 'shipping container', 'shopping cart', 'sleigh',
    'space shuttle', 'spacecraft', 'speedboat', 'sports car', 'steam engine',
    'steam locomotive', 'steam train', 'subway', 'suv', 'tank', 'taxi',
    'tour bus', 'tow truck', 'tractor', 'trailer truck', 'train', 'train car',
    'tricycle', 'trolley', 'trolly', 'truck', 'van', 'vehicle', 'wagon',
    'warship', 'wheelchair',
    
    # Roles, Occupation, and Directed Actions - All people/roles have clear front/back orientation
    'actor', 'adult', 'airman', 'angel', 'artist', 'astronaut', 'astronomer',
    'athlete', 'author', 'baby', 'ballerina', 'barber', 'barista', 'bartender',
    'baseball pitcher', 'baseball player', 'basketball player', 'bassist',
    'beekeeper', 'biker', 'boxer', 'boy', 'bride', 'bridesmaid', 'brunette',
    'bus driver', 'businessman', 'captain', 'carpenter', 'cashier', 'champion',
    'chef', 'child', 'client', 'coach', 'college student', 'commander',
    'commuter', 'competitor', 'conductor', 'construction worker', 'cook',
    'country artist', 'couple', 'cowboy', 'customer', 'dancer', 'daughter',
    'defender', 'diver', 'dj', 'doctor', 'drummer', 'engineer', 'father',
    'fireman', 'fisherman', 'football coach', 'football player', 'girl',
    'goalkeeper', 'golfer', 'grandfather', 'grandmother', 'groom',
    'guitarist', 'gymnast', 'hiker', 'hip hop artist',
    'ice hockey player', 'jockey', 'journalist', 'judge', 'leader', 'man',
    'mascot', 'mechanic', 'metal artist', 'miner', 'monk', 'mother',
    'motorcycle racer', 'motorcyclist', 'mountain biker', 'musician', 'officer',
    'paramedic', 'paratrooper', 'participant', 'passenger', 'pedestrian',
    'person', 'photographer', 'pianist', 'pilot', 'pirate', 'player', 'police',
    'politician', 'pop artist', 'preacher', 'princess', 'prophet', 'protester',
    'putin', 'rapper', 'referee', 'rider', 'rock artist', 'rock climber',
    'rugby player', 'runner', 'sailor', 'santa claus', 'saxophonist', 'scientist',
    'scout', 'sculptor', 'shepherd', 'shopper', 'singer', 'skateboarder',
    'skier', 'snowboarder', 'soldier', 'spectator', 'staff', 'street artist',
    'street vendor', 'student', 'superhero', 'surfer', 'swimmer', 'tattoo artist',
    'teacher', 'technician', 'toddler', 'tour guide', 'tourist', 'trainer',
    'twin', 'vendor', 'vet', 'violinist', 'volunteer', 'waiter', 'warrior',
    'witch', 'woman', 'worker', 'wrestler', 'zombie',
    
    # Animals - Biological (Animals/Body Parts) material cluster
    # All animals have clear front/back orientation (facing direction)
    'alpaca', 'anemone', 'ant', 'antelope', 'armadillo', 'bald eagle', 'barn owl',
    'bee', 'beetle', 'bengal tiger', 'bison', 'blackbird', 'blue jay', 'bluebird',
    'brown bear', 'buffalo', 'bull', 'bulldog', 'bumblebee', 'bunny', 'butterfly',
    'calf', 'camel', 'cardinal', 'cat', 'cattle', 'chameleon', 'chihuahua',
    'chimpanzee', 'chipmunk', 'cock', 'cockatoo', 'cormorant', 'cow', 'coyote',
    'cricket', 'crocodile', 'crow', 'cub', 'dachshund', 'dalmatian', 'deco bear',
    'deer', 'dinosaur', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle',
    'egret', 'elephant', 'falcon', 'fawn', 'flamingo', 'fly', 'fox', 'frog',
    'garden spider', 'gecko', 'german shepherd', 'giraffe', 'goat', 'golden retriever',
    'goldfish', 'goose', 'gorilla', 'ground squirrel', 'guinea pig', 'gull',
    'hamster', 'hen', 'heron', 'hippo', 'hornbill', 'horse', 'horseback',
    'hummingbird', 'humpback whale', 'husky', 'hyaena', 'iguana', 'impala',
    'jellyfish', 'kangaroo', 'koi', 'labrador', 'ladybird', 'lamb', 'lion',
    'lizard', 'macaque', 'macaw', 'mallard duck', 'monkey', 'moose', 'moth',
    'ostrich', 'otter', 'owl', 'panda', 'parrot', 'pelican', 'penguin',
    'pig', 'pigeon', 'polar bear', 'police dog', 'pony', 'poodle', 'puffin', 'pug',
    'python', 'ram', 'rat', 'red panda', 'reindeer', 'rhinoceros', 'robin',
    'salmon', 'scorpion', 'sea lion', 'sea turtle', 'seal', 'shark', 'sheep',
    'shrimp', 'snail', 'snake', 'sparrow', 'spider', 'squirrel', 'squirrel monkey',
    'stork', 'swallowtail butterfly', 'swan', 'tortoise', 'turkey', 'turtle',
    'walrus', 'wasp', 'water buffalo', 'whale', 'zebra',
    
    # Additional objects with clear orientation (from benchmark analysis)
    # Furniture - Seating (only objects with clear orientation)
    'chair', 'armchair',
    # Electronics & Equipment
    'laptop', 'microphone', 'clock',
    # Clothing & Accessories (when worn/has orientation)
    'glasses', 'baseball hat', 'hat', 'sandal',
    # People - Poses/States
    'sit', 'laugh', 'relax',
    # People - Groups
    # People - Costumes
    'costume', 'cosplay', 'halloween costume',
    # People - Faces
    'face'
}

# Structural/architectural objects to filter out from QA generation
# These are parts of scene structures (walls, floors, ceilings, etc.), not standalone objects
STRUCTURAL_OBJECTS_TO_FILTER = {
    'ac vent', 'air vent', 'angled roof', 'angled roof corner', 'angled roof wall',
    'arch', 'arch trim', 'archway', 'asphalt', 'backdrop', 'background', 'background building',
    'background wall', 'balcony', 'balcony support bracket', 'bar bottom', 'bar column',
    'bar counter', 'bar top', 'base', 'base reception', 'baseboard', 'beam', 'beam sill',
    'beam small', 'beam small angle', "bean",'blockout module', 'bottom molding', 'bp', 'brick',
    'brick frame', 'brick glass', 'broad frame', 'building', 'building case',
    'building material', 'building plaque', 'building window', 'cable', 'cable duct',
    'cable element', 'cable rail', 'cable rail holder', 'cable row', 'carpet divider',
    'ceiling', 'ceiling box', 'ceiling construction', 'ceiling decor', 'ceiling fix',
    'ceiling item', 'ceiling lamp', 'ceiling light', 'ceiling louver', 'ceiling module',
    'ceiling molding', 'ceiling objects', 'ceiling ornament', 'ceiling pillar',
    'ceiling planks corner', 'ceiling rack', 'ceiling radiator', 'ceiling top',
    'ceiling trim', 'ceiling vent', 'ceiling windows', 'ceiling wood', 'ceramic floor',
    'column', 'column base', 'column head', 'computer cables', 'concrete roof', 'corner',
    'corner pillar', 'cornice', 'corridor', 'counter corner', 'counter wall',
    'counter wooden corner', 'coving', 'curb', 'curbstone', 'curbstone drawer',
    'decorative wall', 'dev backgrounds', 'dev backgrounds backdrop simple',
    'dev backgrounds cylinder', 'dev backgrounds dome',
    'dev backgrounds pano curved closed', 'dev backgrounds pano curved open',
    'dev backgrounds pano simple', 'dev backgrounds sphere',
    'dev backgrounds u shape closed', 'dev backgrounds u shape open', 'divider', 'door',
    'door arch', 'door cap', 'door cut', 'door frame', 'door glass', 'door handle',
    'door metal plate', 'door molding', 'door opening', 'door rail', 'door shutter',
    'door sign', 'door support', 'door wall', 'door window', 'doorway',
    'doorway operating room', 'doorway wide door', 'double door', 'duct', 'duct end',
    'duct end vertical', 'employee door', 'entrance door', 'environment', 'exhaust fan',
    'exterior corner wall', 'exterior door', 'exterior small window', 'exterior wall',
    'exterior window', 'facade balcony', 'facade corner wall', 'facade door',
    'facade window', 'fire door', 'floor', 'floor and stairs', 'floor board',
    'floor decor', 'floor fixture', 'floor glass', 'floor info board', 'floor lamp',
    'floor lantern', 'floor mat', 'floor module', 'floor planks corner', 'floor pool',
    'floor separator', 'floor tile', 'floor trim', 'floorlock', 'footway', 'foundation',
    'front corner', 'gallery', 'gate door', 'glass partition', 'glass partition door',
    'ground', 'ground pillar', 'hallway', 'hallway side', 'handrail', 'handrail glass',
    'hdri sphere', 'hvac', 'hvac duct', 'individual door', 'inner corner wall',
    'inner structure', 'inner wall', 'interior painted wall', 'interior white wall',
    'kitchen door', 'level floor', 'mall', 'mid roof', 'molding', 'new wall', 'oac duct',
    'operating room door', 'outdoor box', 'outdoor platform', 'outdoor staircase',
    'outdoor wall', 'outer structure', 'outer wall', 'outer walls', 'outside',
    'outside trim', 'outside wall', 'panel', 'panorama', 'park grass', 'park walkway',
    'partition', 'partition door', 'passageway bottom', 'passageway bottom side',
    'passageway wall', 'pavement', 'paving', 'pillar', 'pillar arch', 'pillar support',
    'pipe', 'pipe a curve', 'pipe b curve', 'pipe b t point', 'pipe connector',
    'pipe fitting', 'pipe fix', 'pipe holder', 'polished concrete', 'rack beam',
    'rack pillar', 'rail bar', 'rail walk', 'rail walk stairs', 'railing',
    'railing curve', 'railing handle end', 'railing stairs', 'road', 'road center',
    'road mesh', 'road part', 'road pivot', 'roof', 'roof beam', 'roof deco',
    'roof deco left', 'roof deco right', 'roof eave', 'roof exterior', 'roof gable',
    'roof interior', 'roof item', 'roof leaf', 'roof light', 'roof onigawara',
    'roof plane', 'roof railing', 'roof railing column', 'roof stairs', 'roof structure',
    'roof tile', 'roof top', 'roof trim', 'roof trim corner', 'roof window', 'roofing',
    'room', 'room door', 'room floor', 'scene mesh', 'screen window', 'side balcony',
    'side walk', 'side walk curve', 'sidewalk', 'sill floor', 'skirting', 'shoji rail',
    'shoji screen',
    'skirting board', 'sky', 'sky light', 'sky sphere', 'skybox', 'skylight', 'skysphere',
    'slim frame', 'small column', 'small stairs', 'stair', 'stair cabinet',
    'stair cabinet door', 'stair handrail', 'stair item', 'stair rail', 'stair railings',
    'stair step', 'staircase door', 'stairs', 'stairs floor', 'stairs railing',
    'stairs wall', 'stone floor', 'streetscape', 'structure', 'supermarket',
    'toilet partition', 'top piece', 'top piece corner', 'top small window', 'trim', 'tile'
    'upper stair concrete', 'vent', 'vent pipe', 'ventilation', 'ventilation end',
    'ventilation grill', 'wall', 'wall board', 'wall bottom strip',
    'wall bottom strip large', 'wall cap', 'wall column strip', 'wall corner',
    'wall corner top block', 'wall cover', 'wall decor', 'wall detail',
    'wall detail frame', 'wall div', 'wall div window', 'wall door', 'wall door hole',
    'wall electrical wire casing', 'wall floor', 'wall fountain', 'wall frame',
    'wall fridge', 'wall half', 'wall half corner', 'wall hook', 'wall hose',
    'wall lamp', 'wall large column strip', 'wall letter', 'wall letter line',
    'wall light', 'wall module', 'wall module brick', 'wall module double door',
    'wall module kitchen', 'wall module short', 'wall module slat', 'wall mounted flag',
    'wall panel', 'wall paneling', 'wall paper', 'wall periodic table poster',
    'wall pillar', 'wall pillar metal', 'wall pipe', 'wall posters large',
    'wall posters small', 'wall shelves', 'wall skirting', 'wall slate', 'wall small',
    'wall stair', 'wall switch', 'wall table', 'wall toilet', 'wall top block',
    'wall top block half', 'wall tops panel', 'wall tops panel half',
    'wall triangle banners', 'wall trim', 'wall trim plastic', 'wall trim wood',
    'wall window', 'wall wire', 'wall wire plug', 'wallboard', 'wallpaper', 'wallposter',
    'walls', 'wide door', 'wide frame', 'window', 'window a', 'window blind',
    'window blinds', 'window cap', 'window covering', 'window cut', 'window divided',
    'window fence', 'window frame', 'window frame pole', 'window glass',
    'window molding', 'window roof', 'window side', 'window sill', 'window slide',
    'window var', 'window wall', 'wires', 'wiring', 'wiring turn', 'wood trim',
    'wood wall', 'wooden beam', 'wooden floor', 'wooden wall'
}


def filter_void_clusters(clusters: List[str], taxonomy_type: str) -> List[str]:
    """Filter out void clusters from the given cluster list"""
    void_list = VOID_CLUSTERS.get(taxonomy_type, [])
    return [cluster for cluster in clusters if cluster not in void_list]


def is_void_cluster(cluster_name: str, taxonomy_type: str) -> bool:
    """Check if a cluster is a void cluster that should be excluded"""
    void_list = VOID_CLUSTERS.get(taxonomy_type, [])
    return cluster_name in void_list


def should_exclude_object_from_qa(object_name: str, taxonomy_utils, taxonomy_types_to_check: List[str] = None) -> bool:
    """
    Check if an object should be excluded from ALL question generation.
    
    This checks if the object is in any void cluster across the specified taxonomy types.
    If taxonomy_types_to_check is None, checks all taxonomy types.
    
    Args:
        object_name: Name of the object to check
        taxonomy_utils: TaxonomyUtils instance (can be None, will return False if None)
        taxonomy_types_to_check: List of taxonomy types to check (e.g., ['function', 'material'])
                                If None, checks all: ['function', 'material', 'affordance', 'shape', 'physical', 'texture']
    
    Returns:
        True if object should be excluded, False otherwise
    """
    if not taxonomy_utils:
        return False
    
    if taxonomy_types_to_check is None:
        taxonomy_types_to_check = ['function', 'material', 'affordance', 'shape', 'physical', 'texture']
    
    for tax_type in taxonomy_types_to_check:
        try:
            clusters = taxonomy_utils.get_object_clusters(object_name, f'final_taxonomy_{tax_type}')
            if not clusters:
                continue
            
            has_non_void = any(not is_void_cluster(cluster, tax_type) for cluster in clusters)
            if has_non_void:
                # At least one valid cluster – do not exclude based on this taxonomy type
                continue
            
            # All clusters for this taxonomy type are void – exclude the object
            if all(is_void_cluster(cluster, tax_type) for cluster in clusters):
                return True
        except Exception:
            # If there's an error checking, continue to next taxonomy type
            continue
    
    return False


