import os, cv2, time, json
import math
import numpy as np
import argparse
np.set_printoptions(suppress=True)

import dataclasses

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.utils.pnp.cuboid_objectron import Cuboid3d

import tools.objectron_eval.objectron.dataset.iou as IoU3D
import tools.objectron_eval.objectron.dataset.box as Box
import tools.objectron_eval.objectron.dataset.metrics_nvidia as metrics

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

# Define names for meta information
CONFIDENCE = "confidence"
TOTAL_FAILS = "fails"
# Define names for errors
KEYPOINT = "2d_keypoint"
LOCATION = "location"
DISTANCE_FROM_CAMERA = "distance"
OBJECT_AZIMUTH = "azimuth"
LOCATION_VERTICAL = "vertical"
LOCATION_LATERAL = "planar"
DIRECTION_ANGLE = "dir_ang"
DIRECTION_LOCATION = "dir_loc"


class ErrorDict(dict):

    error_max = { # Maximum errors. Used when no prediction is made
            KEYPOINT: 0.1,
            LOCATION: 2.,
            DISTANCE_FROM_CAMERA: 2.,
            OBJECT_AZIMUTH: 90., 
            LOCATION_VERTICAL: 2., 
            LOCATION_LATERAL: 2.,
            DIRECTION_ANGLE: 45.,
            DIRECTION_LOCATION: 4.,
            }
    ap_max = {
            KEYPOINT: -1,
            LOCATION: 0.3,
            DISTANCE_FROM_CAMERA: 0.3,
            OBJECT_AZIMUTH: 30, 
            LOCATION_VERTICAL: 0.3, 
            LOCATION_LATERAL: 0.3,
            DIRECTION_ANGLE: 15,
            DIRECTION_LOCATION: 0.3,
            }
    ap_threshold = {
            KEYPOINT: -1,
            LOCATION: 0.15,
            DISTANCE_FROM_CAMERA: 0.10,
            OBJECT_AZIMUTH: 10, 
            LOCATION_VERTICAL: 0.10, 
            LOCATION_LATERAL: 0.10,
            DIRECTION_ANGLE: 2,
            DIRECTION_LOCATION: 0.10,
            }
    def __init__(self):
        self.metadict = {
                CONFIDENCE: [[]],
                TOTAL_FAILS: [[]],
                }
        self.total_fails = 0

    def serialize(self, filename):
        self_dict = {}
        for key in self.keys():
            self_dict[key] = self[key]

        d = {
                'total_fails': self.total_fails,
                "dict": self_dict,
                "metadict": self.metadict,
            }
        np.savez(f"{filename}.npz", **d)

    @staticmethod
    def deserialize(filename):
        d = np.load(f"{filename}.npz", allow_pickle=True)
        ret = ErrorDict()
        for key in d['dict'].item():
            ret[key] = d['dict'].item()[key]
        ret.metadict = d['metadict'].item()
        ret.total_fails = d['total_fails'].item()
        return ret


    def failed(self, idx):
        self.append_meta(TOTAL_FAILS, idx)
        self.total_fails += 1

    def get_max(self, name):
        return self.ap_max[name]



    def get_threshold(self, name):
        return self.ap_threshold[name]

    def append_max(self, name):
        if name == KEYPOINT:
            error = [ (0, self.error_max[KEYPOINT]) for _ in range(8) ]
        else:
            error = self.error_max[name]
        self.append(name, error)

    def append_meta(self, name, meta):
        if name in self:
            raise Exception(f"Forbidden meta key {name}.")
        if name in self.metadict:
            self.metadict[name][-1].append(meta)
        else:
            raise Exception(f"Meta key {name} not valid!")
        print(f"META {name}:", meta)


    def append(self, name, error):
        if name == KEYPOINT: # convert kp errors with conf -1 to max_error
            error = np.array(error)
            mask, = np.where(error[:,0] == -1)
            if len(mask) > 0:
                error[mask,0] = 0
                error[mask,1] = self.error_max[name]
            error = error.tolist()
        if name in self.metadict:
            raise Exception(f"Forbidden key {name}.")
        if name in self:
            self[name][-1].append(error)
        else:
            self[name] = [[error]]

        if False and isinstance(error, np.ndarray) or isinstance(error, list):
            # list of errors will be a list of conf, error values
            mean_error = np.array([ i[1] for i in error if i[0] != -1 ]).mean()
            print(f"{name} mean:", mean_error)
        else:
            print(f"{name}:", error)

    def next(self):
        for key in self.keys():
            self[key].append([])
        for key in self.metadict.keys():
            self.metadict[key].append([])

    def numpyize(self):
        for key in self.metadict.keys():
            self.metadict[key] = np.array([ np.array(i) for i in self.metadict[key][:-1]], dtype=object)
        for key in self.keys():
            print(key, len(self[key]))
            for e in self[key]:
                print(len(e))
            self[key] = np.array([ np.array(i) for i in self[key][:-1]], dtype=object)

                
def combine_transform(rmat, tvec):
    return np.vstack((
        np.hstack((rmat, np.array(tvec)[np.newaxis].T)),
        np.array([0.,0.,0.,1.])))

def transform(pts, t):
    pts = np.hstack((pts, np.ones((pts.shape[0],1)))) # to homogeneous
    pts = pts @ t.T
    pts = pts[:,:3] / pts[:,3:4]
    return pts

def draw_cuboid(img, kps, color, thickness):
    cuboid_order = np.array([
            [0,1], [0,2], [1,3], [2,3],
            [0,4], [2,6], [1,5], [3,7],
            [4,5], [4,6], [5,7], [6,7]
            ])
    plt = kps[cuboid_order,:].reshape(-1,2,2)
    img = cv2.polylines(img, plt.astype(int), False, color, thickness)
    # img = cv2.line(img, tuple(kps[1].astype(int)), tuple(kps[2].astype(int)), (0,255,0), 1)

    # img = cv2.drawMarker(img, tuple(kps[1].astype(int)), (0,255,0), cv2.MARKER_CROSS, 20,5)
    return img

def get_lateral_error(pred, gt):

    pred_dir = np.tan(pred[1]/-pred[2])
    gt_dir = np.tan(gt[1]/-gt[2])

    angle_error = np.rad2deg(np.abs(pred_dir - gt_dir))
    distance_error = np.abs(pred[1] - gt[1])

    return angle_error, distance_error


def get_azimuth_error(T1, T2):
    """
    Get azimuth error between two transformations. 
    We define azimuth as rotation around the vertical axis in T1 coordinate space.
    T1 : camera to ground truth
    T2 : camera to prediction
    """
    T = T1 @ np.linalg.inv(T2)

    # rot_offset = (90-img_idx)
    rot_euler = np.rad2deg(R.from_matrix(T[:3,:3]).as_euler("ZXY"))
    rot_error = rot_euler[2] % 180
    if rot_error > 90:
        rot_error -= 180

    return np.abs(rot_error)

def get_keypoint_error(kps, kps_conf, gt, w, h):
    """
    Get mean keypoint error over detections in a single image.
    Missing detections are disregarded. Error is normalized to image size
    Uses L2 distance
    """
    global pause

    print(kps)
    
    mask, = np.where(kps_conf != -1)
    masked = kps[mask]
    masked_gt = gt[mask]

    # Reorder indices for 180 degree symmetry
    symmetry_order = [5,4,7,6,1,0,3,2]
    sym = np.array(symmetry_order)[mask].argsort() # apply mask to sym indices

    # Calculate mean distance between pred and gt

    # Calculate distances for both symmetries
    dist = np.stack((masked - masked_gt, masked[sym] - masked_gt), axis=2)
    dist = np.linalg.norm(dist, axis=1)

    use_sym = np.argmin(np.mean(dist, axis=0))

    if use_sym:
        dist = np.linalg.norm((kps - gt[symmetry_order]) / [w,h], axis=1)
    else:
        dist = np.linalg.norm((kps - gt) / [w,h], axis=1)

    return np.vstack((kps_conf, dist)).T

def merge_keypoint_error(errors):
    kpes = []
    n_kpes = 0
    for kpe in errors:
        kpe = kpe.reshape(-1,2)
        n_kpes += kpe.shape[0]
        kpes.append(np.mean(kpe[np.where(kpe[:,0] != -1),1]))
    mean_keypoint_error = np.mean(kpes)

    return mean_keypoint_error, n_kpes





def compute_ap(errs, confs, max_error, greater=False):
    NUM_BINS = 21
    thresholds = np.linspace(0.0, max_error, num=NUM_BINS)
    ap = metrics.AveragePrecision(NUM_BINS)


    # for confs, errs in zip(confidences, errors):
    hitmiss = metrics.HitMiss(thresholds)
    for conf, err in np.vstack((confs, errs)).T:
        hitmiss.record_hit_miss([err, conf], greater=greater)
    ap.append(hitmiss, len(confs))

    ap.compute_ap_curve()
    return np.vstack((thresholds, ap.aps)).T

def create_report(report_name, dataset_ids, errors, hang=False):
    fails = errors.total_fails
    confs = errors.metadict[CONFIDENCE]
    keypoint_errors = errors[KEYPOINT]
    location_errors = errors[LOCATION]
    azimuth_errors = errors[OBJECT_AZIMUTH]

    # __import__('pdb').set_trace()

    num_preds = np.sum([ len(i) for i in keypoint_errors ])

    print("total fails:", fails)
    print("total preds:", num_preds)
    total = fails + num_preds
    print("Total:", total)

    # Calculate total keypoint error:
    mean_keypoint_error, num_keypoints = merge_keypoint_error(keypoint_errors)

    keypoint_detection_ratio = num_keypoints / (total * 8)
    print("Mean keypoint error:", round(mean_keypoint_error, 2))
    print("Keypoint detection ratio:", round(keypoint_detection_ratio, 2))

    aps = {}
    for key in errors.keys():
        max_value = errors.get_max(key)
        if max_value == -1:
            continue
        
        aps[key] = np.array([ compute_ap(errors[key][i], confs[i], max_value) for i in range(len(dataset_ids)) ])

        # Concatenate APs into one column per dataset
        ap = aps[key]
        aps[key] = np.hstack((ap[0,:,0:1], ap[:,:,1:].reshape((len(dataset_ids), -1)).T))


    # __import__('pdb').set_trace()
    # exit()

    # location_aps = compute_ap(stats['location_errors'], confs, 0.3)
    # azimuth_aps = compute_ap(azimuth_errors, confs, 20)

    
    for i, key in enumerate(aps.keys()):
        plt.subplot(2, math.ceil(len(aps)/2), i+1)
        plt.title(key)
        plt.xlabel("Threshold")
        plt.ylabel("Average Precision")
        plt.ylim((0,1))
        for d in range(len(dataset_ids)):
            plt.plot(aps[key][:,0], aps[key][:,1+d], label=f'model_{dataset_ids[d]}')
        plt.grid()
        plt.legend()
    plt.suptitle(report_name)
    if hang:
        plt.show()
    else:
        plt.draw()

    for key in aps:
        csv_array = np.round(aps[key],5)
        headers = ["confidence"]
        for ds in dataset_ids:
            headers.append(f"{key}_{ds}")
        csv_array = np.vstack((headers, csv_array))
        np.savetxt(f'./reports/{report_name}_{key}.csv', csv_array, delimiter=' ', fmt='%s')

    exit()
    
    csv_array = [ np.round(aps[LOCATION][0,:],5) ]
    headers = [CONFIDENCE, LOCATION]
    for key in aps.keys():
        csv_array.append(aps[key][1,:])
        headers.append(key)
    csv_array = np.array(csv_array).T

    csv_array = np.vstack((headers, csv_array))


    # location_aps = aps['location_ap']
    # azimuth_aps = aps['azimuth_ap']
    # # Create CSV array
    # csv_array = np.array([np.round(location_aps[0,:],5),
    #                       location_aps[1,:],
    #                       azimuth_aps[1,:]
    #                       ]).T
    # Add headers
    # csv_array = np.vstack((["confidence", "location_ap", "azimuth_ap"], csv_array))

    np.savetxt('report_test.csv', csv_array, delimiter=' ', fmt='%s')
    # return


    for i, key in enumerate(aps.keys()):
        print(key)
        plt.subplot(2, math.ceil(len(aps)/2), i+1)
        plt.title(key)
        plt.xlabel("Threshold")
        plt.ylabel("AP")
        plt.plot(aps[key][0,:], aps[key][1,:])

    # plt.subplot(121)
    # plt.title("Location AP")
    # plt.xlabel("Location Threshold")
    # plt.ylabel("AP")
    # plt.plot(location_aps[0,:], location_aps[1,:])
    # plt.subplot(122)
    # plt.title("Azimuth AP")
    # plt.xlabel("Azimuth Threshold")
    # plt.ylabel("AP")
    # plt.plot(azimuth_aps[0,:], azimuth_aps[1,:])
    plt.show()



    exit()

from_opengl = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,-1]])
to_opengl = np.linalg.inv(from_opengl)
pause=False

def validate(opt, models, dataset_dir, datasets, step=1, scale_mult=None, save=True):
    global pause
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    errors = ErrorDict()

    total_predictions = 0

    K, w, h = get_intrinsics(f"{dataset_dir}/val_env_{datasets[0][0]}_take_{datasets[0][1]}")
    opt.cam_intrinsic = K
    meta = {}
    meta['camera_matrix'] = opt.cam_intrinsic

    for model in models:
        opt.load_model = model

        Detector = detector_factory[opt.task]
        detector = Detector(opt)

        detector.pause = False

        for env, take in datasets:
            dataset = f"{dataset_dir}/val_env_{env}_take_{take}/annotated"
        
            if not os.path.isdir(dataset):
                print(f"Dataset {dataset} not found!")
                exit()
                
            ls = os.listdir(dataset)
            img_names = []
            for file_name in sorted(ls):
                body, ext = os.path.splitext(file_name)
                if ext.lower() == '.png':
                    img_name = f"{dataset}/{body}{ext}"
                    img_names.append(img_name)

            img_idx = 0
            while img_idx < len(img_names):

                # if img_idx > 50:
                #     break

                image_name = img_names[img_idx]

                ret = detector.run(image_name, meta_inp=meta)

                with open(os.path.splitext(image_name)[0] + ".json", 'r') as f:
                    ann = json.load(f)
                gt = ann['objects'][0]

                img = cv2.imread(image_name)

                total_predictions += 1

                # get GT transformation w.r.t camera in opencv coordinates
                loc_gt = np.array(gt['location'])
                rot_gt = R.from_quat(gt['quaternion_xyzw']).as_matrix()
                rot90 = combine_transform(R.from_euler("ZYX",np.deg2rad([0,90,0])).as_matrix(), np.zeros(3))
                T_gt = combine_transform(rot_gt, loc_gt)
                T_gt = from_opengl @ T_gt @ rot90

                cp_gt = np.array(gt['projected_cuboid']).reshape(-1,2)[0,:]
                kps_gt = kps_gt = np.array(gt['projected_cuboid']).reshape(-1,2)[1:,:]


                # draw GT
                img = draw_cuboid(img, kps_gt, (0,255,0), 2)

                has_pred = 'boxes' in ret and len(ret['boxes']) > 0
                if has_pred:
                    pred = ret['boxes'][0]
                    box_point_2d, box_point_3d, _, _, result_ori = pred

                    confidence = result_ori['score']
                    kps_confidence = np.array(result_ori['kps_score'])

                    loc_pred = np.array(result_ori['location'])
                    if scale_mult is not None:
                        loc_pred *= scale_mult
                    rot_pred = result_ori['quaternion_xyzw']
                    rot_pred = R.from_quat(rot_pred).as_matrix()

                    cp_pred = np.array(result_ori['ct'])
                    kps_pred = result_ori['kps'].reshape(-1,2)
                    cub_pred = box_point_2d[1:,:] * [w, h]

                    T_pred = combine_transform(rot_pred, loc_pred)

                    T_pred = from_opengl @ T_pred
                    T_pred2gt = T_gt @ np.linalg.inv(T_pred)

                    # box_gt = Box.Box(box_point_3d_gt)
                    # box_pred = Box.Box(box_point_3d)
                    # iou = IoU3D.IoU(box_pred, box_gt)

                    # Calculate errors
                    kps_errors = get_keypoint_error(kps_pred, kps_confidence, kps_gt, w, h)
                    distance_error = np.abs(np.linalg.norm(loc_pred) - np.linalg.norm(loc_gt))
                    location_error = np.linalg.norm(loc_pred - loc_gt)
                    azimuth_error = get_azimuth_error(T_gt, T_pred)
                    latang_error, latdist_error = get_lateral_error(loc_pred, loc_gt)

                    loc_in_gt = transform(np.array([loc_pred]), np.linalg.inv(T_gt) @ to_opengl)
                    height_error = np.abs(loc_in_gt[0,1])
                    planar_error = np.linalg.norm(loc_in_gt[0,[0,2]])

                    mean_kp_error = np.array([ e[1] for e in kps_errors if e[0] != -1 ]).mean()
                    if mean_kp_error > 1000:
                        print("Mean keypoint error over 1000!")
                        pause=True

                    print(kps_errors)

                    errors.append(KEYPOINT, kps_errors)
                    errors.append(OBJECT_AZIMUTH, azimuth_error)
                    errors.append(DIRECTION_ANGLE, latang_error)
                    errors.append(LOCATION, location_error)
                    errors.append(DISTANCE_FROM_CAMERA, distance_error)
                    errors.append(LOCATION_VERTICAL, height_error)
                    errors.append(LOCATION_LATERAL, planar_error)
                    errors.append(DIRECTION_LOCATION, latdist_error)

                    errors.append_meta(CONFIDENCE, confidence)


                    # Draw detections
                    img = draw_cuboid(img, cub_pred, (255,0,0), 2)
                    for i, kp in enumerate(kps_pred.squeeze()):
                        col = (255,0,0)
                        img = cv2.drawMarker(img, tuple(kp.astype(int)), col, cv2.MARKER_CROSS, 25, 3)
                        img = cv2.putText(img, str(i+1), tuple(kp.astype(int)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                        img = cv2.putText(img, str(round(kps_confidence[i],2)), tuple(kp.astype(int) - [0,25]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
                else: # no prediction
                    errors.failed(img_idx)

                    errors.append_max(KEYPOINT)
                    errors.append_max(OBJECT_AZIMUTH)
                    errors.append_max(DIRECTION_ANGLE)
                    errors.append_max(LOCATION)
                    errors.append_max(DISTANCE_FROM_CAMERA)
                    errors.append_max(LOCATION_VERTICAL)
                    errors.append_max(LOCATION_LATERAL)
                    errors.append_max(DIRECTION_LOCATION)

                    errors.append_meta(CONFIDENCE, 0)


                cv2.imshow("Preview", img)
                key = cv2.waitKey(0 if pause else 1)
                if pause:
                    pause=False

                if key == ord('p'):
                    dir = -step
                elif key == 27:
                    break
                    # exit()
                else:
                    dir = step

            
                img_idx += dir
                # print(img_idx)
        errors.next()

    cv2.destroyAllWindows()

    errors.numpyize()
    
    if save:
        errors.serialize(report_name)
    # np.savez(f"{report_name}.npz", **errors)

    return errors
    create_report(errors)

    exit()


    azimuth_threshold = 10 # degrees
    location_threshold = 0.15 # m (15cm)
    azimuth_success = azimuth_errors[:,1] < azimuth_threshold
    location_success = location_errors[:,1] < location_threshold 

    print(keypoint_errors)
    print(location_errors)
    print(azimuth_errors)

    print(len(azimuth_errors), len(location_errors), len(keypoint_errors))
    print(total_predictions)

    # Calculate total mean keypoint error


    print("Total mean keypoint error:", round(np.mean(keypoint_errors),4))
    print("% of correct azimuths:", round(np.mean(azimuth_success),4))
    print("% of correct locations:", round(np.mean(location_success),4))





def get_intrinsics(dataset):

    with open(f"{dataset}/metadata.json", 'r') as f:
        meta = json.load(f)

    intrinsics = meta['streams']['cam2']['intrinsics']

    fov = 90
    w = intrinsics['width']
    h = intrinsics['height']
    fu = (w / 2) / np.tan(np.deg2rad(fov / 2))

    K = np.eye(3)
    K[0,0] = fu
    K[1,1] = fu
    K[0,2] = w / 2.
    K[1,2] = h / 2. + 70

    return K, w, h

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-d', '--dataset', type=int, required=True)
    parser.add_argument('--displacement', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--multi_object', action='store_true')
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--prefix', default=None, type=str)
    parser.add_argument('-d', '--datasets', default=None, type=str)
    args = parser.parse_args()

    opt = opts().parser.parse_args([])

    opt.c = 'pallet'
    opt.task = 'object_pose'
    if args.datasets is not None:
        dataset_ids = [ int(ds) for ds in args.datasets.split(',') ]
    else:
        # dataset_ids = [1,2,3,4,5,6,7]
        # dataset_ids = [1,2,3,4]
        dataset_ids = [1]
    suffix  = f"{'hp' if args.displacement else 'nohp'}"
    suffix += f"_{'scale' if args.scale else 'noscale'}"
    suffix += f"_{'multi' if args.multi_object else 'single'}"
    if args.prefix is not None:
        models = [ f'../exp/object_pose/pallet_final3_ds_{dataset_id}_{args.prefix}_{suffix}/pallet_last.pth' for dataset_id in dataset_ids ]
    else:
        models = [ f'../exp/object_pose/pallet_final3_ds_{dataset_id}_{suffix}/pallet_last.pth' for dataset_id in dataset_ids ]
    if args.prefix is not None:
        report_name = f"{args.prefix}_{suffix}"
    else:
        report_name = f"{suffix}"

    # report_name = f"4_{'newval_' if args.newval else ''}{suffix}"

    opt.nms = True
    opt.arch = 'dlav1_34'
    opt.obj_scale = args.scale

    validation_set_dir = f"/media/localadmin/0c21d63f-0916-4325-a37c-33263ee1cba7/home/olli/Data/Validation"
    validation_sets = [
            (1,2),
            (5,4),
            (6,1),
            (6,2),
            (6,3),
            (7,1),
            (7,2),
            (7,3),
            (8,1),
            (9,1),
            (9,2),
            (9,3),
            (10,1),
            (10,2),
            ]


    opt.rep_mode = 1 if args.displacement else 4 # Heatmap only
    opt.debug = 0
    opt.K = 100 if args.multi_object else 1

    opt.use_pnp = True

    opt = opts().parse(opt)
    opt = opts().init(opt)

    if args.load:
        errors = ErrorDict.deserialize(report_name)
        create_report(report_name, dataset_ids, errors, hang=args.load)
    else:
        errors = validate(opt, models, validation_set_dir, validation_sets, step=3, save=not args.nosave, scale_mult=None if not args.scale else 0.144)
        create_report(report_name, dataset_ids, errors, hang=args.load)

