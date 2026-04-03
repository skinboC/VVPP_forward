import pymeshlab
from tqdm import tqdm
import trimesh
import numpy as np
import os
import glob


def get_folder_size(folder_path):
    """
    计算指定文件夹的总大小。
    """
    total_size = 0
    # os.walk 遍历目录树
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            # 拼接完整的文件路径
            fp = os.path.join(dirpath, f)
            # 检查是否为符号链接，如果是则跳过以避免重复计算或无限循环
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size



def mesh_preprocessing(input_folder, output_folder):
    """
    加载指定文件夹下的所有OBJ文件，然后，对他们进行如下可能的操作：
    - 将模型的中心移动到原点，缩放和平移到[-1, 1]的区域内
    - 通过isotropic explicit remeshing重新网格化
    
    然后将处理后的文件保存到另一个文件夹。
    参数:
    input_folder (str): 包含原始OBJ文件的文件夹路径。
    output_folder (str): 用于保存处理后OBJ文件的文件夹路径。
    """
    # 检查并创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")

    # 获取所有OBJ文件的路径
    obj_files = glob.glob(os.path.join(input_folder, '*.obj'))

    if not obj_files:
        print(f"在文件夹 '{input_folder}' 中未找到任何.obj文件。")
        return

    print(f"找到 {len(obj_files)} 个.obj文件。开始处理...")

    for obj_path in tqdm(obj_files, desc="处理进度"):
        try:
            # 加载模型
            mesh = trimesh.load_mesh(obj_path, process=False)
            
            # 1. 计算模型的边界框中心并平移至原点
            center = mesh.bounds.mean(axis=0)
            translation_matrix = trimesh.transformations.translation_matrix(-center)
            mesh.apply_transform(translation_matrix)

            # 2. 计算缩放比例并进行缩放
            max_extent = np.max(mesh.extents)
            scale = 2.0 / max_extent
            scale *= 0.95  # 留出5%的边距
            scale_matrix = trimesh.transformations.scale_matrix(scale)
            mesh.apply_transform(scale_matrix)

            # 构建输出文件路径
            base_filename = os.path.basename(obj_path)
            output_path = os.path.join(output_folder, base_filename)

            # 导出处理后的模型
            mesh.export(output_path)
            
            # 3. 重新网格化
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(output_path)
            ms.apply_filter('meshing_isotropic_explicit_remeshing', iterations=6, targetlen=pymeshlab.PercentageValue(2.0))
            os.remove(output_path)  # 删除之前的文件
            ms.save_current_mesh(output_path)

          #  print(f"已处理并保存: {output_path}")

        except Exception as e:
            print(f"处理文件 '{obj_path}' 时发生错误: {e}")

    print("所有文件处理完成！")
    input_folder_disk_usage = get_folder_size(input_folder) / (1024 * 1024)
    output_folder_disk_usage = get_folder_size(output_folder) / (1024 * 1024)
    print(f"输入文件夹大小: {input_folder_disk_usage:.2f} MB")
    print(f"输出文件夹大小: {output_folder_disk_usage:.2f} MB")


if __name__ == '__main__':
    # --- 配置 ---
    # 设置包含你的OBJ文件的源文件夹路径
    source_directory = '../data/eigen_mesh'
    # 设置你想要保存处理后文件的目标文件夹路径
    destination_directory = '../data/coarse_eigen_mesh'
    # -------------

    # 将相对路径转换为绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_directory = os.path.abspath(os.path.join(current_dir, source_directory))
    destination_directory = os.path.abspath(os.path.join(current_dir, destination_directory))

    if not os.path.isdir(source_directory):
        print(f"错误：源文件夹不存在: {source_directory}")
    else:
        mesh_preprocessing(source_directory, destination_directory)
        print("\n处理完成！")