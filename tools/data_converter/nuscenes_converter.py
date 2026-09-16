#수정: VLA-FULL — SparseDrive(swc-17/SparseDrive) tools/data_converter/nuscenes_converter.py 포팅.
#  NuScenesMiniDataset 가 읽는 스키마(token / cams[sensor2lidar_*, cam_intrinsic, data_path] /
#  lidar2ego_* / ego2global_* / timestamp / lidar_path / scene_token / gt_boxes / gt_names /
#  gt_velocity / valid_flag / gt_agent_fut_* / gt_ego_fut_*)를 그대로 생성한다.
#  원본 대비 유일한 변경: map_annos(NuscMapExtractor) 제거 — HiP-AD 에 map_utils 미포팅이고
#  현행 VLA-v1 ablation 은 det-only(task_select=["det"]) 라 map GT 를 소비하지 않는다.
#  map 을 켤 때 NuscMapExtractor 를 포팅한 뒤 이 부분을 복원할 것.
import os
import math
import argparse
from os import path as osp

import numpy as np
from pyquaternion import Quaternion

import mmcv

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global

#수정: Path B — map_annos 복원. NuscMapExtractor 는 create_nuscenes_infos 안에서 lazy import.
MAP_CLASSES = ('ped_crossing', 'divider', 'boundary')

def geom2anno(map_geoms):
    """get_map_geom 결과(category→geoms) → {label_idx: [np.array(line)]}."""
    vectors = {}
    for cls, geom_list in map_geoms.items():
        if cls in MAP_CLASSES:
            label = MAP_CLASSES.index(cls)
            vectors[label] = []
            for geom in geom_list:
                vectors[label].append(np.array(geom.coords))
    return vectors

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          roi_size=(30, 60)):  #수정: Path B — map BEV range
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    #수정: Path B — nuScenes map GT 추출기 (lazy import, plugin 경로)
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))
    from mmdet3d_plugin.datasets.map_utils.nuscmap_extractor import NuscMapExtractor
    nusc_map_extractor = NuscMapExtractor(root_path, roi_size)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        out_path = osp.join(out_path, 'mini')
    else:
        raise ValueError('unknown')
    os.makedirs(out_path, exist_ok=True)

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps,
        nusc_map_extractor=nusc_map_extractor)  #수정: Path B

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)

def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=12,
                         ego_fut_ts=6,
                         nusc_map_extractor=None):  #수정: Path B
    """Generate the train/val infos from the raw data."""
    train_nusc_infos = []
    val_nusc_infos = []
    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx

    predict_helper = PredictHelper(nusc)
    for sample in mmcv.track_iter_progress(nusc.sample):
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'map_location': map_location,
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        #수정: Path B — map_annos 복원. lidar2global pose 로 ego 주변 vectorized map 추출.
        if nusc_map_extractor is not None:
            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = l2e_r_mat
            lidar2ego[:3, 3] = np.array(l2e_t)
            ego2global = np.eye(4)
            ego2global[:3, :3] = e2g_r_mat
            ego2global[:3, 3] = np.array(e2g_t)
            lidar2global = ego2global @ lidar2ego
            translation = list(lidar2global[:3, 3])
            rotation = list(Quaternion(matrix=lidar2global).q)
            map_geoms = nusc_map_extractor.get_map_geom(map_location, translation, rotation)
            info['map_annos'] = geom2anno(map_geoms)

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            # object detection annos: boxes (locs, dims, yaw, velocity), names and valid flags
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'

            # object tracking annos: instance_ids
            instance_inds = [nusc.getind('instance', anno['instance_token'])
                             for anno in annotations]

            # motion prediction annos: future trajectories offset in lidar frame and valid mask
            num_box = len(boxes)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
            gt_fut_masks = np.zeros((num_box, fut_ts))
            for i, anno in enumerate(annotations):
                instance_token = anno['instance_token']
                fut_traj_local = predict_helper.get_future_for_agent(
                    instance_token,
                    sample['token'],
                    seconds=fut_ts/2,
                    in_agent_frame=True
                )
                if fut_traj_local.shape[0] > 0:
                    box = boxes[i]
                    trans = box.center
                    rot = Quaternion(matrix=box.rotation_matrix)
                    fut_traj_scene = convert_local_coords_to_global(fut_traj_local, trans, rot)
                    valid_step = fut_traj_scene.shape[0]
                    gt_fut_trajs[i, 0] = fut_traj_scene[0] - box.center[:2]
                    gt_fut_trajs[i, 1:valid_step] = fut_traj_scene[1:] - fut_traj_scene[:-1]
                    gt_fut_masks[i, :valid_step] = 1

            # motion planning annos: future trajectories offset in lidar frame and valid mask
            ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3))
            ego_fut_masks = np.zeros((ego_fut_ts + 1))
            sample_cur = sample
            ego_status = get_ego_status(nusc, nusc_can_bus, sample_cur)
            for i in range(ego_fut_ts + 1):
                pose_mat = get_global_sensor_pose(sample_cur, nusc)
                ego_fut_trajs[i] = pose_mat[:3, 3]
                ego_fut_masks[i] = 1
                if sample_cur['next'] == '':
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])
            # global to ego
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # ego to lidar
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # drive command according to final fut step offset
            if ego_fut_trajs[-1][0] >= 2:
                command = np.array([1, 0, 0])  # Turn Right
            elif ego_fut_trajs[-1][0] <= -2:
                command = np.array([0, 1, 0])  # Turn Left
            else:
                command = np.array([0, 0, 1])  # Go Straight
            # get offset
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['instance_inds'] = instance_inds
            info['gt_agent_fut_trajs'] = gt_fut_trajs.astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = command.astype(np.float32)
            info['ego_status'] = ego_status

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def get_ego_status(nusc, nusc_can_bus, sample):
    ego_status = []
    ref_scene = nusc.get("scene", sample['scene_token'])
    try:
        pose_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'pose')
        steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
        pose_uts = [msg['utime'] for msg in pose_msgs]
        steer_uts = [msg['utime'] for msg in steer_msgs]
        ref_utime = sample['timestamp']
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]
        ego_status.extend(pose_data["accel"])  # acceleration in ego vehicle frame, m/s/s
        ego_status.extend(pose_data["rotation_rate"])  # angular velocity in ego vehicle frame, rad/s
        ego_status.extend(pose_data["vel"])  # velocity in ego vehicle frame, m/s
        ego_status.append(steer_data["value"])  # steering angle, positive: left turn, negative: right turn
    except:
        ego_status = [0] * 10

    return np.array(ego_status).astype(np.float32)

def get_global_sensor_pose(rec, nusc):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    cs_record = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

    ego2global = transform_matrix(pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False)
    sensor2ego = transform_matrix(cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False)
    pose = ego2global.dot(sensor2ego)

    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR."""
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset."""
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
    parser.add_argument(
        '--root-path', type=str, default='./data/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--canbus', type=str, default='./data/nuscenes',
        help='specify the root path of nuScenes canbus')
    parser.add_argument(
        '--version', type=str, default='v1.0-trainval', required=False,
        help='specify the dataset version')
    parser.add_argument(
        '--max-sweeps', type=int, default=10, required=False,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--out-dir', type=str, default='./data/infos', required=False,
        help='output dir of info pkl')
    parser.add_argument('--extra-tag', type=str, default='nuscenes')
    return parser.parse_args()


if __name__ == '__main__':
    #수정: HiP-AD 단독 실행용 — projects plugin import 불필요(NuscMapExtractor 제거).
    args = parse_args()
    assert args.dataset == 'nuscenes'
    nuscenes_data_prep(
        root_path=args.root_path,
        can_bus_root_path=args.canbus,
        info_prefix=args.extra_tag,
        version=args.version,
        out_dir=args.out_dir,
        max_sweeps=args.max_sweeps)
