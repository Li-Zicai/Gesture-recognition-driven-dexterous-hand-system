#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：检查手势识别系统的所有组件
"""

import os
import json
import numpy as np
from pathlib import Path

def check_file_exists(filename: str, description: str) -> bool:
    """检查文件是否存在"""
    exists = os.path.exists(filename)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filename}")
    return exists

def check_model_checkpoint(model_path: str) -> bool:
    """检查模型文件的有效性"""
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'input_size', 'hidden_size', 'num_layers', 'num_classes']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        
        if missing_keys:
            print(f"✗ 模型缺少必要字段: {missing_keys}")
            return False
        
        print(f"✓ 模型检查通过")
        print(f"  - input_size: {checkpoint['input_size']}")
        print(f"  - hidden_size: {checkpoint['hidden_size']}")
        print(f"  - num_layers: {checkpoint['num_layers']}")
        print(f"  - num_classes: {checkpoint['num_classes']}")
        return True
    except ImportError:
        print("✗ PyTorch 未安装")
        return False
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

def check_label_encoder(encoder_path: str) -> dict:
    """检查标签编码器"""
    try:
        encoder = np.load(encoder_path, allow_pickle=True).item()
        print(f"✓ 标签编码器检查通过")
        print(f"  - 手势类别: {list(encoder.keys())}")
        print(f"  - 映射: {encoder}")
        return encoder
    except Exception as e:
        print(f"✗ 标签编码器加载失败: {e}")
        return {}

def check_gesture_actions(actions_path: str, label_encoder: dict) -> bool:
    """检查手势动作文件的有效性"""
    try:
        with open(actions_path, 'r') as f:
            actions_data = json.load(f)
        
        # 提取有效的手势动作
        valid_actions = {
            k: v["joints"] for k, v in actions_data.items()
            if isinstance(v, dict) and "joints" in v
        }
        
        print(f"✓ 手势动作文件检查通过")
        print(f"  - 定义的手势: {list(valid_actions.keys())}")
        
        # 检查是否所有标签都有对应的动作
        missing_actions = set(label_encoder.keys()) - set(valid_actions.keys())
        if missing_actions:
            print(f"⚠ 警告: 以下手势没有定义动作: {missing_actions}")
        
        # 检查每个动作的关节数
        for gesture_name, joints in valid_actions.items():
            if len(joints) != 17:
                print(f"✗ {gesture_name}: 关节数 {len(joints)}, 应该是 17")
                return False
        
        print(f"  - 每个手势都定义了 17 个关节 ✓")
        return True
    except Exception as e:
        print(f"✗ 手势动作文件读取失败: {e}")
        return False

def check_template_file(template_path: str) -> bool:
    """检查模板文件"""
    try:
        with open(template_path, 'r') as f:
            templates = json.load(f)
        
        print(f"✓ 模板文件检查通过")
        print(f"  - 定义的模板: {list(templates.keys())}")
        return True
    except Exception as e:
        print(f"⚠ 模板文件可选，但加载失败: {e}")
        return True  # 可选文件

def main():
    print("\n" + "="*60)
    print("手势识别系统验证")
    print("="*60 + "\n")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 1. 检查文件存在性
    print("1️⃣ 文件检查:")
    print("-" * 40)
    model_exists = check_file_exists("gesture_model.pth", "模型文件")
    encoder_exists = check_file_exists("label_encoder.npy", "标签编码器")
    actions_exists = check_file_exists("gesture_actions.json", "手势动作文件")
    templates_exists = check_file_exists("gesture_templates.json", "模板文件")
    print()
    
    # 2. 检查模型
    print("2️⃣ 模型检查:")
    print("-" * 40)
    if model_exists:
        model_ok = check_model_checkpoint("gesture_model.pth")
    else:
        model_ok = False
    print()
    
    # 3. 检查标签编码器
    print("3️⃣ 标签编码器检查:")
    print("-" * 40)
    if encoder_exists:
        label_encoder = check_label_encoder("label_encoder.npy")
    else:
        label_encoder = {}
    print()
    
    # 4. 检查手势动作
    print("4️⃣ 手势动作文件检查:")
    print("-" * 40)
    if actions_exists:
        actions_ok = check_gesture_actions("gesture_actions.json", label_encoder)
    else:
        actions_ok = False
    print()
    
    # 5. 检查模板
    print("5️⃣ 模板文件检查:")
    print("-" * 40)
    if templates_exists:
        templates_ok = check_template_file("gesture_templates.json")
    else:
        templates_ok = True  # 可选
    print()
    
    # 总结
    print("=" * 60)
    print("✅ 系统就绪!" if (model_exists and encoder_exists and actions_exists) else "⚠️ 系统缺少必要文件")
    print("=" * 60)
    
    # 建议
    print("\n📋 后续步骤:")
    if model_exists and encoder_exists and actions_exists:
        print("1. 启动服务器: python hand_netserver.py")
        print("2. 观察日志中的手势识别输出")
        print("3. 如需调整手势动作，编辑 gesture_actions.json")
    else:
        if not model_exists:
            print("- 训练模型: python train_gesture_model.py")
        if not actions_exists:
            print("- 创建手势动作文件: 编辑 gesture_actions.json")
    
    print()

if __name__ == "__main__":
    main()
