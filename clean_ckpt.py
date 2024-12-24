import os
import glob
import shutil

def clean_checkpoints(base_dir):
    # 查找所有包含checkpoints的实验目录
    exp_dirs = glob.glob(os.path.join(base_dir, "**", "checkpoints"), recursive=True)
    
    for ckpt_dir in exp_dirs:
        print(f"\n处理目录: {ckpt_dir}")
        
        # 获取所有checkpoint文件夹
        ckpts = []
        for item in os.listdir(ckpt_dir):
            try:
                # 尝试将文件夹名转换为数字
                ckpt_num = int(item)
                ckpts.append(ckpt_num)
            except ValueError:
                continue
        
        if not ckpts:
            print("未找到checkpoint文件夹")
            continue
            
        # 排序找出最后一个checkpoint
        ckpts.sort()
        last_ckpt = ckpts[-1]
        
        # 确定要保留的checkpoint
        keep_ckpts = set()
        # 添加5000的倍数
        for ckpt in ckpts:
            if ckpt % 500 == 0:
                keep_ckpts.add(ckpt)
        # 添加最后一个checkpoint
        keep_ckpts.add(last_ckpt)

        # 确保1不在keep_ckpts中
        if 1 in keep_ckpts:
            keep_ckpts.remove(1)
        
        # 删除不需要保留的checkpoint
        for ckpt in ckpts:
            if ckpt not in keep_ckpts:
                ckpt_path = os.path.join(ckpt_dir, str(ckpt).zfill(7))
                if os.path.exists(ckpt_path):
                    print(f"删除: {ckpt_path}")
                    shutil.rmtree(ckpt_path)
        
        print(f"保留的checkpoint: {sorted(list(keep_ckpts))}")

if __name__ == "__main__":
    # 设置基础目录路径
    # base_dirs = ["/mnt/hwfile/alpha_vl/duruoyi/in_context_results_v6"]
    # for base_dir in base_dirs:
    #     clean_checkpoints(base_dir)

    base_dirs = ["/mnt/hwfile/alpha_vl/duruoyi/lumina_image3_results"]
    for base_dir in base_dirs:
        clean_checkpoints(base_dir)