from __future__ import print_function
from pprint import pprint
import traceback
import sys
import shutil
import openmesh as om
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import pymesh
import torch.optim as optim
from pytorch_points.misc import logger
from pytorch_points.network.geo_operations import mean_value_coordinates_3D, edge_vertex_indices
from pytorch_points.utils.pc_utils import load, save_ply, save_pts, center_bounding_box
from pytorch_points.utils.geometry_utils import read_trimesh, write_trimesh, build_gemm, Mesh, get_edge_points, generatePolygon
from pytorch_points.utils.pytorch_utils import weights_init, check_values, save_network, load_network, save_grad, saved_variables, \
    clamp_gradient_norm, linear_loss_weight, tolerating_collate, clamp_gradient, fix_network_parameters
import losses
import networks
from common import loadInitCage, build_dataset, crisscross_input, log_outputs, deform_with_MVC
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, \
    PointNet2ClsSsg, PointConvDensityClsSsg
from adv_utils import LogitsAdvLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


def test(net=None, save_subdir="test"):
    opt.phase = "test"
    opt.num_point = 1024
    dataset = build_dataset(opt)

    if opt.dim == 3:
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1,1,-1).cuda()]

    if net is None:
        # network
        net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size,
                               template_vertices=cage_V_t, template_faces=init_cage_Fs[-1],
                               ).cuda()
        net.eval()
        load_network(net, opt.ckpt)
    else:
        net.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                             collate_fn=tolerating_collate,
                                             num_workers=0,
                                             worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    test_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)

    with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data = dataset.uncollate(data)

                # Blending logic
                if opt.blend_style:
                    num_alpha = 4
                    blend_alpha = torch.linspace(0, 1, steps=num_alpha, dtype=torch.float32).cuda().reshape(num_alpha, 1)
                    data["source_shape"] = data["source_shape"].expand(num_alpha, -1, -1).contiguous()
                    data["target_shape"] = data["target_shape"].expand(num_alpha, -1, -1).contiguous()
                else:
                    blend_alpha = 1.0

                data["alpha"] = blend_alpha

                # Running the network
                source_shape_t = data["source_shape"].transpose(1,2).contiguous().detach()
                target_shape_t = data["target_shape"].transpose(1,2).contiguous().detach()

                outputs = net(source_shape_t, target_shape_t, blend_alpha)
                deformed = outputs["deformed"]

                s_filename = os.path.splitext(data["source_file"][0])[0]
                t_filename = os.path.splitext(data["target_file"][0])[0]

                log_str = "{}/{} {}-{} ".format(i, len(dataloader), s_filename, t_filename)
                print(log_str)
                f.write(log_str+"\n")

                # Save outputs (for visual inspection or further processing)
                for b in range(deformed.shape[0]):
                    # save to "pts" for rendering
                    save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sa.pts".format(s_filename,t_filename)), data["source_shape"][b].detach().cpu())
                    save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sab-{}.pts".format(s_filename,t_filename, b)), deformed[0].detach().cpu())

                    save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sb.pts".format(s_filename,t_filename)), data["target_shape"][0].detach().cpu())

                    outputs["cage"][b] = center_bounding_box(outputs["cage"][b])[0]
                    outputs["new_cage"][b] = center_bounding_box(outputs["new_cage"][b])[0]
                    pymesh.save_mesh_raw(
                        os.path.join(opt.log_dir, save_subdir, "{}-{}-cage1-{}.ply".format(s_filename, t_filename, b)),
                        outputs["cage"][b].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)
                    pymesh.save_mesh_raw(
                        os.path.join(opt.log_dir, save_subdir, "{}-{}-cage2-{}.ply".format(s_filename, t_filename, b)),
                        outputs["new_cage"][b].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)




import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_point_cloud(pc, output_dir, file_name, point_size=1):
    """ 포인트 클라우드를 시각화하여 이미지로 저장하는 함수 """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=point_size, c='b', marker='o')
    
    # 축 제한 설정 (-1~1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])


    # 이미지 저장
    output_path = os.path.join(output_dir, f"{file_name}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def train():
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                             collate_fn=tolerating_collate, num_workers=2, 
                                             worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    if opt.dim == 3:
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        print(opt.template)
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        cage_edge_points_list = []
        cage_edges_list = []
        for F in init_cage_Fs:
            mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
            build_gemm(mesh, F[0])
            cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
            cage_edge_points_list.append(cage_edge_points)
            cage_edges_list = [edge_vertex_indices(F[0])]
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1, 1, -1).cuda()]

    # NetworkFull 초기화
    net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size, template_vertices=cage_V_t, template_faces=init_cage_Fs[-1]).cuda()

    net.apply(weights_init)
    if opt.ckpt:
        load_network(net, opt.ckpt)

    all_losses = losses.AllLosses(opt)
    optimizer = torch.optim.Adam([{"params": net.encoder.parameters()},
                                  {"params": net.nd_decoder.parameters()},
                                  {"params": net.merger.parameters()}], lr=opt.lr)

    if opt.full_net:
        optimizer.add_param_group({'params': net.nc_decoder.parameters(), 'lr': 0.1*opt.lr})
    if opt.optimize_template:
        optimizer.add_param_group({'params': net.template_vertices, 'lr': opt.lr})

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs * 0.4), gamma=0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1)

    net.train()
    start_epoch = 0
    t = 0
    steps_C = 20
    steps_D = 20

    source_shape_saved = False
    all_source_shapes = []

    # npz 저장을 위한 새로운 폴더 생성
    visualization_dir = os.path.join(opt.log_dir, 'visualization')
    os.makedirs(visualization_dir, exist_ok=True)

    os.makedirs(opt.log_dir, exist_ok=True)
    shutil.copy2(__file__, opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "networks.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "losses.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "datasets.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "common.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "option.py"), opt.log_dir)

    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write(str(net) + "\n")

    log_interval = max(len(dataloader) // 5, 50)
    save_interval = max(opt.nepochs // 10, 1)
    running_avg_loss = -1

    with torch.autograd.detect_anomaly():
        if opt.epoch:
            start_epoch = opt.epoch % opt.nepochs
            t += start_epoch * len(dataloader)

        for epoch in range(start_epoch, opt.nepochs):
            for t_epoch, data in enumerate(dataloader):
                warming_up = epoch < opt.warmup_epochs
                optimize_C = (t % (steps_C + steps_D)) > steps_D

                data = dataset.uncollate(data)
                data = crisscross_input(data)

                source_shape, target_shape = data["source_shape"], data["target_shape"]

                # source_shape 데이터 저장
                all_source_shapes.append(source_shape[:8].cpu().numpy())

                if opt.blend_style:
                    blend_alpha = torch.rand((source_shape.shape[0], 1), dtype=torch.float32).to(device=source_shape.device)
                else:
                    blend_alpha = 1.0
                data["alpha"] = blend_alpha

                optimizer.zero_grad()
                source_shape_t = source_shape.transpose(1, 2)
                target_shape_t = target_shape.transpose(1, 2)
                outputs = net(source_shape_t, target_shape_t, data["alpha"])

                current_loss = all_losses(data, outputs, epoch)
                loss_sum = torch.sum(torch.stack([v for v in current_loss.values()], dim=0))

                if running_avg_loss < 0:
                    running_avg_loss = loss_sum
                else:
                    running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss) / (t + 1)

                if (t % log_interval == 0) or (loss_sum > 5 * running_avg_loss):
                    log_str = f"warming up {warming_up} e {epoch:03d} t {t:05d}: " + ", ".join([f"{k} {v.mean().item():.3g}" for k, v in current_loss.items()])
                    print(log_str)
                    log_file.write(log_str + "\n")
                    log_outputs(opt, t, outputs, data)

                if loss_sum > 100 * running_avg_loss:
                    logger.info(f"loss ({loss_sum}) > 5*running_average_loss ({5*running_avg_loss}). Skip without update.")
                    torch.cuda.empty_cache()
                    continue

                loss_sum.backward()
                clamp_gradient(net, 0.1)
                optimizer.step()

                if (t + 1) % 500 == 0:
                    save_network(net, opt.log_dir, network_label="net", epoch_label="latest")

                t += 1

            if (epoch + 1) % save_interval == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label=epoch)

            scheduler.step()
            if opt.eval:
                try:
                    test(net=net, save_subdir=f"epoch_{epoch}")
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                    logger.warn("Failed to run test", str(e))

    # source_shape 데이터 한번만 저장 및 시각화
    if not source_shape_saved and len(all_source_shapes) > 0:
        # 시각화
        for i, shape in enumerate(all_source_shapes[0]):
            visualize_point_cloud(shape, visualization_dir, f"source_shape_{i}", point_size=1)

        # npz 저장
        np.savez(os.path.join(opt.log_dir, 'source_shape_data.npz'), test_pc=np.concatenate(all_source_shapes, axis=0))
        source_shape_saved = True

    log_file.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    test(net=net)


#=========================================================================================================================================================================
class_mapping = {
    "02691156": 0,  # Airplane
    "02773838": 1,  # Bag
    "02954340": 2,  # Cap
    "02958343": 3,  # Car
    "03001627": 4,  # Chair
    "03261776": 5,  # Earphone
    "03467517": 6,  # Guitar
    "03624134": 7,  # Knife
    "03636649": 8,  # Lamp
    "03642806": 9,  # Laptop
    "03790512": 10, # Motorbike
    "03797390": 11, # Mug
    "03948459": 12, # Pistol
    "04099429": 13, # Rocket
    "04225987": 14, # Skateboard
    "04379243": 15  # Table
}

# 경로 리스트를 입력받아 각 파일의 라벨을 반환하는 함수
def get_label_from_file_id(file_id, base_dir):
    """
    주어진 파일명이 어느 클래스에 속하는지 탐색하여 해당 클래스 라벨을 반환하는 함수
    """
    # 클래스 폴더 순회
    for class_id, label in class_mapping.items():
        # 해당 클래스 폴더 경로
        class_dir = os.path.join(base_dir, class_id)
        
        # 파일이 해당 클래스 폴더에 존재하는지 확인
        file_path = os.path.join(class_dir, f"{file_id}.txt")
        
        if os.path.exists(file_path):
            return label
    
    # 파일을 찾지 못한 경우
    raise FileNotFoundError(f"File {file_id}.txt not found in any class folders.")


def process_nested_file_ids(file_ids, base_dir):
    """
    Processes file IDs, which can be either nested lists or flat lists,
    and returns the corresponding labels.
    """
    labels = []
    if isinstance(file_ids, list):
        if all(isinstance(f, list) for f in file_ids):
            # Nested list
            for file_id_list in file_ids:
                for file_id in file_id_list:
                    try:
                        label = get_label_from_file_id(file_id, base_dir)
                        labels.append(label)
                    except FileNotFoundError as e:
                        print(e)
        else:
            # Flat list
            for file_id in file_ids:
                try:
                    label = get_label_from_file_id(file_id, base_dir)
                    labels.append(label)
                except FileNotFoundError as e:
                    print(e)
    else:
        # Single file ID
        try:
            label = get_label_from_file_id(file_ids, base_dir)
            labels.append(label)
        except FileNotFoundError as e:
            print(e)
    return labels

def optimize_cage_and_face(cage, point, distance=0.4, iters=1000, step=0.01):
    """
    입력:
    - cage: (B, 3, N2) 크기의 배치별 케이지 꼭짓점
    - face: (B, N3, 3) 크기의 배치별 케이지 면 (인덱스)
    - point: (B, 3, N1) 크기의 배치별 포인트
    - distance: 각 포인트가 케이지에서 떨어져야 하는 최소 거리
    - iters: 최적화를 진행할 반복 횟수
    - step: 케이지 업데이트 스텝 크기
    
    출력:
    - 최적화된 cage와 face
    """
    B, _, N2 = cage.shape
    B, _, N1 = point.shape

    # 케이지를 복사하여 확장 (배치 크기를 고려)
    cage_expanded = cage.clone()

    for _ in range(iters):
        # 케이지와 포인트 사이의 벡터 계산
        diff = cage_expanded[:, :, :, None] - point[:, :, None, :]  # (B, 3, N2, N1)

        # 현재 포인트와 케이지 사이의 거리 계산
        current_distance = torch.norm(diff, dim=1)  # (B, N2, N1)

        # 각 케이지 점에서 가장 가까운 포인트와의 최소 거리
        min_distance, _ = torch.min(current_distance, dim=2)  # (B, N2)

        # 최소 거리보다 떨어진 경우에만 업데이트 수행
        do_update = (min_distance > distance).float().unsqueeze(1)  # (B, 1, N2)

        # 이동 방향 벡터 계산
        vector = torch.sum(diff / (current_distance.unsqueeze(1) + 1e-8), dim=3)  # (B, 3, N2)

        # 케이지 위치 업데이트 (거리보다 작은 점들에 대해만)
        cage_expanded = cage_expanded + step * vector * do_update

    return cage_expanded

def adv_train():
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                             collate_fn=tolerating_collate,
                                             num_workers=2, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
    
    if opt.dim == 3:
        # cage (1,N,3)
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        cage_edge_points_list = []
        cage_edges_list = []
        for F in init_cage_Fs:
            mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
            build_gemm(mesh, F[0])
            cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
            cage_edge_points_list.append(cage_edge_points)
            cage_edges_list = [edge_vertex_indices(F[0])]
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1,1,-1).cuda()]

    # network
    net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size,
                               template_vertices=cage_V_t, template_faces=init_cage_Fs[-1],
                               ).cuda()
    
    #===================================================================================
    #####################################load model###################################
    #model = PointNetCls(k=16, feature_transform=False).cuda()
    #state_dict = torch.load("/workspace/deep_cage/trained_clssifier/BEST_model195_acc_0.9777.pth", map_location='cpu')
    model = DGCNN(1024, 20, output_channels=16).cuda()
    state_dict = torch.load("/workspace/deep_cage/trained_clssifier/model90_acc_0.9805_loss_0.0341_lr_0.000593.pth", map_location='cpu')    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    #===================================================================================
    save_dir = "/workspace/deep_cage/log/deep_cage_log_80"
    ckpt_name ='net_latest.pth'
    class_name = "cage_deformer_3d-" + opt.name
    opt.ckpt = os.path.join(save_dir, class_name, ckpt_name)

    net.apply(weights_init)
    load_network(net, opt.ckpt)

    # net.nd_decoder 외 모든 네트워크의 파라미터를 업데이트하지 않도록 설정
    for name, param in net.named_parameters():
        if "nd_decoder" not in name:
            param.requires_grad = False

    all_losses = losses.AllLosses(opt)
    
    # optimizer: net.nd_decoder 파라미터만 업데이트
    optimizer = torch.optim.Adam([
        {"params": net.nd_decoder.parameters()}], lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(int(opt.nepochs*0.4), 1), gamma=0.1, last_epoch=-1)


    # train
    net.train()
    start_epoch = 0
    t = 0

    steps_C = 20
    steps_D = 20

    # train
    os.makedirs(opt.log_dir, exist_ok=True)
    shutil.copy2(__file__, opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "networks.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "losses.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "datasets.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "common.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "option.py"), opt.log_dir)

    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write(str(net)+"\n")

    log_interval = max(len(dataloader)//5, 50)
    save_interval = max(opt.nepochs//10, 1)
    running_avg_loss = -1

    with torch.autograd.detect_anomaly():
        if opt.epoch:
            start_epoch = opt.epoch % opt.nepochs
            t += start_epoch*len(dataloader)

        for epoch in range(start_epoch, opt.nepochs):
            for t_epoch, data in enumerate(dataloader):
                warming_up = epoch < opt.warmup_epochs
                progress = t_epoch/len(dataloader)+epoch
                optimize_C = (t % (steps_C+steps_D)) > steps_D

                ############# get data ###########
                data = dataset.uncollate(data)
                #data = crisscross_input(data)
                if opt.dim == 3:
                    data["cage_edge_points"] = cage_edge_points_list[-1]
                    data["cage_edges"] = cage_edges_list[-1]
                source_shape, target_shape = data["source_shape"], data["target_shape"]

                source_label = torch.tensor(process_nested_file_ids(data["source_file"], "/workspace/deep_cage/data/shape_data"), dtype=torch.long).cuda()
                target_label = torch.tensor(process_nested_file_ids(data["target_file"], "/workspace/deep_cage/data/shape_data"), dtype=torch.long).cuda()

                ############# blending ############
                if opt.blend_style:
                    blend_alpha = torch.rand((source_shape.shape[0], 1), dtype=torch.float32).to(device=source_shape.device)
                else:
                    blend_alpha = 1.0
                data["alpha"] = blend_alpha

                ############# run network ###########
                optimizer.zero_grad()
                source_shape_t = source_shape.transpose(1,2).contiguous()
                target_shape_t = target_shape.transpose(1,2).contiguous()

                outputs = net(source_shape_t, target_shape_t, data["alpha"])

                logits = model(outputs["deformed"].transpose(1,2))  # [B, num_classes]

                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]

                preds = torch.argmax(logits, dim=-1)

                ############# get losses ###########
                current_loss = all_losses(data, outputs, progress)
                adv_loss = adv_func(logits, source_label).mean()
                current_loss["adv_loss"] = adv_loss * 1
                del current_loss["MSE"]
                del current_loss["SCD"] 
                
                loss_sum = torch.sum(torch.stack([v for v in current_loss.values()], dim=0))
                if running_avg_loss < 0:
                    running_avg_loss = loss_sum
                else:
                    running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss)/(t+1)

                if (t % log_interval == 0) or (loss_sum > 5 * running_avg_loss):
                    # 공격 성공 여부를 계산
                    attack_success_count = (preds != source_label).sum().item()  # preds와 source_label이 다른 경우의 수
                    log_str = "warming up {} e {:03d} t {:05d}: {} | Attack Success Count: {}".format(
                        warming_up, epoch, t,
                        ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()]),
                        attack_success_count
                    )
                    print(log_str)

                if loss_sum > 100*running_avg_loss:
                    logger.info("loss ({}) > 5*running_average_loss ({}). Skip without update.".format(loss_sum, 5*running_avg_loss))
                    torch.cuda.empty_cache()
                    continue

                loss_sum.backward()
                clamp_gradient(net, 0.1)
                optimizer.step()

                if (t + 1) % 500 == 0:
                    save_network(net, opt.log_dir, network_label="net", epoch_label="latest")

                t += 1

            if (epoch + 1) % save_interval == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label=epoch)

            scheduler.step()
            if opt.eval:
                try:
                    test(net=net, save_subdir="epoch_{}".format(epoch))
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                    logger.warn("Failed to run test", str(e))

    log_file.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    adv_test(net=net)

def adv_test(net=None, save_subdir="test"):
    opt.phase = "test"
    opt.num_point = 1024
    dataset = build_dataset(opt)

    if opt.dim == 3:
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor(
            [(x, y) for x, y in init_cage_V], dtype=torch.float
        ).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        init_cage_Fs = [
            torch.arange(opt.cage_deg, dtype=torch.int64).view(1, 1, -1).cuda()
        ]

    if net is None:
        # Network
        net = networks.NetworkFull(
            opt,
            dim=opt.dim,
            bottleneck_size=opt.bottleneck_size,
            template_vertices=cage_V_t,
            template_faces=init_cage_Fs[-1],
        ).cuda()
        net.eval()
        load_network(net, opt.ckpt)
    else:
        net.eval()

    # Load the classifier model
    #model = PointNetCls(k=16, feature_transform=False).cuda()
    #state_dict = torch.load("/workspace/deep_cage/trained_clssifier/BEST_model195_acc_0.9777.pth", map_location='cpu')
    model = DGCNN(1024, 20, output_channels=16).cuda()
    state_dict = torch.load("/workspace/deep_cage/trained_clssifier/model90_acc_0.9805_loss_0.0341_lr_0.000593.pth", map_location='cpu') 
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Eliminate 'module.' in keys if present
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=tolerating_collate,
        num_workers=0,
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id),
    )

    test_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)

    with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data = dataset.uncollate(data)

                # Blending logic
                if opt.blend_style:
                    num_alpha = 4
                    blend_alpha = (
                        torch.linspace(0, 1, steps=num_alpha, dtype=torch.float32)
                        .cuda()
                        .reshape(num_alpha, 1)
                    )
                    data["source_shape"] = (
                        data["source_shape"]
                        .expand(num_alpha, -1, -1)
                        .contiguous()
                    )
                    data["target_shape"] = (
                        data["target_shape"]
                        .expand(num_alpha, -1, -1)
                        .contiguous()
                    )
                else:
                    blend_alpha = 1.0

                data["alpha"] = blend_alpha

                # Obtain labels
                source_label = torch.tensor(
                    process_nested_file_ids(
                        data["source_file"], "/workspace/deep_cage/data/shape_data"
                    ),
                    dtype=torch.long,
                ).cuda()
                target_label = torch.tensor(
                    process_nested_file_ids(
                        data["target_file"], "/workspace/deep_cage/data/shape_data"
                    ),
                    dtype=torch.long,
                ).cuda()

                # Running the network
                source_shape_t = (
                    data["source_shape"].transpose(1, 2).contiguous().detach()
                )
                target_shape_t = (
                    data["target_shape"].transpose(1, 2).contiguous().detach()
                )

                outputs = net(source_shape_t, target_shape_t, blend_alpha)
                deformed = outputs["deformed"]

                # Get predictions from classifier
                logits = model(deformed.transpose(1, 2))
                if isinstance(logits, tuple):  # For models like PointNet
                    logits = logits[0]
                preds = torch.argmax(logits, dim=-1)

                s_filename = os.path.splitext(data["source_file"][0])[0]
                t_filename = os.path.splitext(data["target_file"][0])[0]

                log_str = "{}/{} {}-{} Predicted: {} True: {}".format(
                    i,
                    len(dataloader),
                    s_filename,
                    t_filename,
                    preds[0].item(),
                    source_label[0].item(),
                )
                print(log_str)
                f.write(log_str + "\n")

                # Save outputs (for visual inspection or further processing)
                for b in range(deformed.shape[0]):
                    deformed = deform_with_MVC(
                        outputs["cage"][b : b + 1],
                        outputs["new_cage"][b : b + 1],
                        outputs["cage_face"],
                        data["source_shape"][b : b + 1],
                    )

                    # Save to "pts" for rendering, including predicted class in filename
                    save_pts(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-Sab-{}-pred{}.pts".format(
                                s_filename, t_filename, b, preds[b].item()
                            ),
                        ),
                        deformed[b].detach().cpu(),
                    )

                    # Optionally, save source and target shapes with true labels
                    save_pts(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-Sa-true{}.pts".format(
                                s_filename, t_filename, source_label[b].item()
                            ),
                        ),
                        data["source_shape"][b].detach().cpu(),
                    )
                    save_pts(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-Sb-true{}.pts".format(
                                s_filename, t_filename, target_label[b].item()
                            ),
                        ),
                        data["target_shape"][b].detach().cpu(),
                    )

                    # Save cages if necessary
                    outputs["cage"][b] = center_bounding_box(outputs["cage"][b])[0]
                    outputs["new_cage"][b] = center_bounding_box(
                        outputs["new_cage"][b]
                    )[0]
                    pymesh.save_mesh_raw(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-cage1-{}.ply".format(s_filename, t_filename, b),
                        ),
                        outputs["cage"][b].detach().cpu(),
                        outputs["cage_face"][0].detach().cpu(),
                        binary=True,
                    )
                    pymesh.save_mesh_raw(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-cage2-{}.ply".format(s_filename, t_filename, b),
                        ),
                        outputs["new_cage"][b].detach().cpu(),
                        outputs["cage_face"][0].detach().cpu(),
                        binary=True,
                    )

import time  # 시간 측정을 위해 time 모듈을 가져옵니다.

def adv_train_feature():
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # 필요에 따라 True로 설정할 수 있습니다.
        drop_last=False,
        collate_fn=tolerating_collate,
        num_workers=2,
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id)
    )

    if opt.dim == 3:
        # 케이지 설정 (1, N, 3)
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        cage_edge_points_list = []
        cage_edges_list = []
        for F in init_cage_Fs:
            mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
            build_gemm(mesh, F[0])
            cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
            cage_edge_points_list.append(cage_edge_points)
            cage_edges_list = [edge_vertex_indices(F[0])]
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor(
            [(x, y) for x, y in init_cage_V], dtype=torch.float
        ).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        init_cage_Fs = [
            torch.arange(opt.cage_deg, dtype=torch.int64).view(1, 1, -1).cuda()
        ]

    # 네트워크 설정
    net = networks.NetworkFull_adv(
        opt,
        dim=opt.dim,
        bottleneck_size=opt.bottleneck_size,
        template_vertices=cage_V_t,
        template_faces=init_cage_Fs[-1],
    ).cuda()

    # ================================================
    # 분류 모델 로드
    #model = PointNetCls(k=16, feature_transform=False).cuda()
    #state_dict = torch.load("/workspace/deep_cage/trained_clssifier/BEST_model195_acc_0.9777.pth", map_location='cpu')
    model = DGCNN(1024, 20, output_channels=16).cuda()
    state_dict = torch.load("/workspace/deep_cage/trained_clssifier/model90_acc_0.9805_loss_0.0341_lr_0.000593.pth", map_location='cpu')   
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Eliminate 'module.' in keys if present
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # ================================================
    save_dir = "/workspace/deep_cage/log/deep_cage_log_80_optimized_cage"
    ckpt_name ='net_latest.pth'
    class_name = "cage_deformer_3d-" + opt.name
    opt.ckpt = os.path.join(save_dir, class_name, ckpt_name)

    net.apply(weights_init)
    load_network(net, opt.ckpt)

    # 네트워크의 모든 파라미터를 업데이트하지 않도록 설정
    for name, param in net.named_parameters():
        param.requires_grad = False

    all_losses = losses.AllLosses(opt)

    # 출력 디렉토리 생성
    os.makedirs(opt.log_dir, exist_ok=True)
    shutil.copy2(__file__, opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "networks.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "losses.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "datasets.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "common.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "option.py"), opt.log_dir)

    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write(str(net) + "\n")

    log_interval = max(len(dataloader) // 5, 1)
    running_avg_loss = -1

    t = 0

    for batch_idx, data in enumerate(dataloader):
        data = dataset.uncollate(data)
        #data = crisscross_input(data)
        if opt.dim == 3:
            data["cage_edge_points"] = cage_edge_points_list[-1]
            data["cage_edges"] = cage_edges_list[-1]
        source_shape, target_shape = data["source_shape"], data["target_shape"]

        source_label = torch.tensor(
            process_nested_file_ids(
                data["source_file"], opt.data_dir
            ),
            dtype=torch.long,
        ).cuda()
        target_label = torch.tensor(
            process_nested_file_ids(
                data["target_file"], opt.data_dir
            ),
            dtype=torch.long,
        ).cuda()

        B = source_shape.shape[0]

        # 각 배치마다 새로운 코드북 생성
        codebook = torch.randn(B, opt.bottleneck_size, 1, device='cuda') * 1e-7
        codebook = nn.Parameter(codebook, requires_grad=True)

        # 코드북에 대한 옵티마이저 생성
        optimizer = torch.optim.Adam([codebook], lr=opt.lr*10)

        net.train()
        # 해당 배치에 대해 opt.nepochs 만큼 학습
        for epoch in range(opt.nepochs):
            optimizer.zero_grad()

            # Blending logic
            if opt.blend_style:
                blend_alpha = torch.rand(
                    (B, 1), dtype=torch.float32
                ).to(device=source_shape.device)
            else:
                blend_alpha = 1.0
            data["alpha"] = blend_alpha

            # 네트워크 실행
            source_shape_t = source_shape.transpose(1, 2).contiguous()
            target_shape_t = target_shape.transpose(1, 2).contiguous()

            outputs = net(
                source_shape_t,
                target_shape_t,
                additional_input=codebook,
                alpha=data["alpha"],
            )

            logits = model(outputs["deformed"].transpose(1, 2))  # [B, num_classes]

            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]

            preds = torch.argmax(logits, dim=-1)

            # 손실 계산
            current_loss = all_losses(data, outputs, epoch)
            adv_loss = adv_func(logits, source_label).mean()
            current_loss["adv_loss"] = adv_loss
            del current_loss["MSE"]
            current_loss["CD"] = torch.tensor(0.0).cuda()

            loss_sum = torch.sum(
                torch.stack([v for v in current_loss.values()], dim=0)
            )

            loss_sum.backward()
            optimizer.step()

            # 로그 출력 (각 손실의 이름과 값을 명확하게 출력)
            if epoch % log_interval == 0:
                attack_success_count = (preds != source_label).sum().item()
                loss_details = ", ".join(
                    ["{}: {:.4f}".format(k, v.mean().item()) for k, v in current_loss.items()]
                )
                log_str = "Batch {:03d} Epoch {:03d}: Total Loss {:.4f} | Losses [{}] | Attack Success Count: {}".format(
                    batch_idx,
                    epoch,
                    loss_sum.item(),
                    loss_details,
                    attack_success_count,
                )
                print(log_str)

        # 학습이 끝난 후 결과 저장
        net.eval()
        with torch.no_grad():
            # 데이터 평탄화: 각 샘플마다 하나의 ID로 설정
            #data["source_file"] = [item for sublist in data["source_file"] for item in sublist]
            #data["target_file"] = [item for sublist in data["target_file"] for item in sublist]

            # 저장할 폴더 생성
            save_subdir = "batch_{:03d}".format(batch_idx)
            save_folder = os.path.join(opt.log_dir, save_subdir)
            os.makedirs(save_folder, exist_ok=True)

            source_shape_t = source_shape.transpose(1, 2).contiguous()
            target_shape_t = target_shape.transpose(1, 2).contiguous()

            outputs = net(
                source_shape_t,
                target_shape_t,
                additional_input=codebook,
                alpha=data["alpha"],
            )

            deformed = outputs["deformed"]

            # 분류 모델로부터 예측값 얻기
            logits = model(deformed.transpose(1, 2))
            if isinstance(logits, tuple):  # PointNet 등
                logits = logits[0]
            preds = torch.argmax(logits, dim=-1)

            # 결과 저장
            for b in range(B):
                # source_file_id와 target_file_id 추출
                s_id = data["source_file"][b]  # 이미 평탄화된 1차원 리스트
                t_id = data["target_file"][b]

                s_filename = s_id
                t_filename = t_id

                # deformed를 center_bounding_box로 중심 정렬
                deformed_b = deformed[b]

                # 변형된 모양 저장
                save_pts(
                    os.path.join(
                        save_folder,
                        "{}-{}-Sab-{}-pred{}.pts".format(
                            s_filename, t_filename, b, preds[b].item()
                        ),
                    ),
                    deformed_b.detach().cpu(),
                )

                # 소스와 타겟 모양 저장
                save_pts(
                    os.path.join(
                        save_folder,
                        "{}-{}-Sa-true{}.pts".format(
                            s_filename, t_filename, source_label[b].item()
                        ),
                    ),
                    data["source_shape"][b].detach().cpu(),
                )
                save_pts(
                    os.path.join(
                        save_folder,
                        "{}-{}-Sb-true{}.pts".format(
                            s_filename, t_filename, target_label[b].item()
                        ),
                    ),
                    data["target_shape"][b].detach().cpu(),
                )

                # 케이지 저장
                outputs["cage"][b] = center_bounding_box(outputs["cage"][b])[0]
                outputs["new_cage"][b] = center_bounding_box(outputs["new_cage"][b])[0]
                pymesh.save_mesh_raw(
                    os.path.join(
                        save_folder,
                        "{}-{}-cage1-{}.ply".format(s_filename, t_filename, b),
                    ),
                    outputs["cage"][b].detach().cpu(),
                    outputs["cage_face"][0].detach().cpu(),
                    binary=True,
                )
                pymesh.save_mesh_raw(
                    os.path.join(
                        save_folder,
                        "{}-{}-cage2-{}.ply".format(s_filename, t_filename, b),
                    ),
                    outputs["new_cage"][b].detach().cpu(),
                    outputs["cage_face"][0].detach().cpu(),
                    binary=True,
                )

        # 다음 배치로 넘어감

    log_file.close()


if __name__ == "__main__":
    from option import BaseOptions
    import datetime
    import os
    parser = BaseOptions()
    opt = parser.parse()

    # reproducability
    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    np.random.seed(24)

    adv_func = LogitsAdvLoss(kappa=1.0)

    if opt.phase == "train" or opt.phase == "adv_train" or opt.phase == "adv_train_feature":
        if opt.ckpt is not None:
            opt.log_dir = os.path.dirname(opt.ckpt)
        else:
            opt.log_dir = os.path.join(opt.log_dir, "-".join(filter(None, [os.path.basename(__file__)[:-3],
                                                                        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                                        opt.name])))       

    else:
        try:
            opt.log_dir = os.path.dirname(opt.ckpt)
        except:
            print("train_victim")

    if opt.phase == "test":
        test(save_subdir=opt.subdir)

    elif opt.phase == "adv_train":
        os.makedirs(opt.log_dir, exist_ok=True)
        log_file = open(os.path.join(opt.log_dir, "adv_training_log.txt"), "a")
        parser.print_options(opt, log_file)
        log_file.close()
        adv_train()
        
    elif opt.phase == "adv_train_feature":
        os.makedirs(opt.log_dir, exist_ok=True)
        log_file = open(os.path.join(opt.log_dir, "adv_training_log.txt"), "a")
        parser.print_options(opt, log_file)
        log_file.close()
        adv_train_feature()

    else:
        os.makedirs(opt.log_dir, exist_ok=True)
        log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
        parser.print_options(opt, log_file)
        log_file.close()
        train()