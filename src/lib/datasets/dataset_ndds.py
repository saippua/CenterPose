from scipy.spatial.transform import Rotation as R
import json
import numpy as np
from tqdm import tqdm
import os
import cv2

def quatmult(q1, q2):
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def get_files(dir):
    files = [ file.split('.')[0] for file in os.listdir(dir) ]
    files = np.unique([ file for file in files if file.isnumeric() ])
    files = [ f"{dir}/{file}" for file in files ]
    print(f"{len(files)} datapoints found in dataset '{dir}'")
    return files

def convert_json(old, cs):

    # This function converts 2D or 3D points from UE coordinate system to CenterPose
    def pc(pt):
        if len(pt) == 2:
            # 2D points should be pixel coordinates (int)
            return [ int(round(pt[0])), int(round(pt[1])) ]
        elif len(pt) == 3:
            # 3D points convert from UEs left handed coordinate system to CenterPose right handed
            return [ pt[1], pt[0], -pt[2] ]

    new = dict()

    w = cs['camera_settings'][0]['intrinsic_settings']['resX']
    h = cs['camera_settings'][0]['intrinsic_settings']['resY']

    camera_data = {
            'camera_projection_matrix': old['camera_data']['projectionMatrix'],
            'camera_view_matrix': old['camera_data']['viewProjectionMatrix'],
            'width': w,
            'height': h,
            'intrinsics': { 
                           'cx': w - cs['camera_settings'][0]['intrinsic_settings']['cx'],
                           'cy': cs['camera_settings'][0]['intrinsic_settings']['cy'],
                           'fx': cs['camera_settings'][0]['intrinsic_settings']['fx'],
                           'fy': cs['camera_settings'][0]['intrinsic_settings']['fy'],
                           },
            'location_world': pc(old['camera_data']['location_worldframe']) + [ 1.0 ],
            'quaternion_world_xyzw': old['camera_data']['quaternion_xyzw_worldframe'],
        }

    objects = [ 
            { 
                'class': 'pallet' if old_obj['class'] == 'pallet_eur' else old_obj['class'],
                'keypoints_3d': [ pc(old_obj['cuboid_centroid']) ] + [ pc(pt) for pt in old_obj['cuboid'] ],
                'location': pc(old_obj['location']) + [ 1.0 ],
                'name': 'pallet_01',
                'projected_cuboid': [ pc(old_obj['projected_cuboid_centroid']) ] + [ pc(pt) for pt in old_obj['projected_cuboid'] ],
                'pose_transform': old_obj['pose_transform'],
                'provenance': 'simod',
                'quaternion_xyzw': old_obj['quaternion_xyzw'],
                'scale': [ 0.8, 0.144, 1.2 ],
                'symmetric': True,

            }
            for old_obj in old['objects'] ]
    new = {
            'camera_data': camera_data,
            'objects': objects,
        }

    ## Rescale coordinates from cm to m
    new['camera_data']['location_world'] = (np.array(new['camera_data']['location_world']) / 100.).tolist()
    for obj in new['objects']:
        obj['pose_transform'][3] = (np.array(obj['pose_transform'][3]) / 100.).tolist()
        obj['location'] = (np.array(obj['location']) / 100.).tolist()

    ## Reorder cuboids
    idx = [0,4,3,1,2,8,7,5,6]
    reorder = lambda a, idx: (np.array(a)[idx]).tolist()

    for obj in new['objects']:
        obj['projected_cuboid'] = reorder(obj['projected_cuboid'], idx)
        obj['keypoints_3d'] = reorder(obj['keypoints_3d'], idx)

    ## Apply rotation offset
    # 90 degree rotation around X and Y axis to convert 
    rot_offset = R.from_euler("ZYX", np.deg2rad([0, 90, 90])).as_quat()[[2,1,0,3]]
    rot_offset[3] *= -1
    for obj in new['objects']:
        # pass
        obj['quaternion_xyzw'] = quatmult(obj['quaternion_xyzw'], rot_offset)[[3,2,0,1]].tolist()
        obj['quaternion_xyzw'][0] *= -1

    return new


def swap_background(fg, bg, mask):
    bg = cv2.resize(bg, fg.shape[:2], interpolation=cv2.INTER_CUBIC)
    np.putmask(fg, 1-mask, bg)

    return fg


def loadimages_ndds(dir, max_count=None):
    imgs = []
    with open(f"{dir}/_object_settings.json") as f_os:
        obj_s = json.loads(f_os.read())

    with open(f"{dir}/_camera_settings.json") as f_cs:
        cam_s = json.loads(f_cs.read())

    print("Dataset contains following objects:")

    mask_ids = []
    for i, o in enumerate(obj_s['exported_objects']):
        print(i+1, "-", o['class'].ljust(15), 'id:', o['segmentation_class_id'])
        mask_ids.append(o['segmentation_class_id'])
    print()

    files = get_files(dir);

    if max_count is not None:
        files = files[:max_count]

    # files = filter(files, o['segmentation_class_id'], 300)

    for i, file in enumerate(files):
        if not os.path.isfile(f"{file}.dope.json"):
            with open(f"{file}.json", 'r') as f:
                old_json = json.loads(f.read())

            new = convert_json(old_json, cam_s)

            with open(f"{file}.dope.json", 'w') as f:
                f.write(json.dumps(new, indent=4))

        mask = cv2.imread(f"{file}.cs.png")
        mask = np.isin(mask, mask_ids).astype(np.float32)
            
        imgs.append((
            f"{file}.png",       # Img filename
            i,                   # Video ID
            0,                   # Frame ID
            f"{file}.dope.json", # Annotation filename
            mask
        ))
    print(f"Loaded {len(imgs)} datapoints")
    return imgs

# loadimages_ndds("/home/olli/Data/Generated/pallet/", 500)
