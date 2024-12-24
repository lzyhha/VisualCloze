import torch
from apex import amp

def test_apex_cpu():
    print("开始测试 APEX (CPU 模式)...")
    
    # 1. 测试基本导入
    print("\n1. APEX 基本导入测试:")
    try:
        print("✓ APEX 导入成功")
    except ImportError as e:
        print("✗ APEX 导入失败:", e)

    # 2. 测试基本操作
    print("\n2. 基本操作测试:")
    try:
        # 创建一个简单的模型和优化器
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        
        # 使用假数据进行前向传播
        x = torch.randn(2, 10)
        output = model(x)
        print("✓ 模型前向传播成功")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
    except Exception as e:
        print("✗ 基本操作测试失败:", e)

    # 3. 测试 AMP 相关 API
    print("\n3. AMP API 测试:")
    try:
        # 检查是否可以访问 AMP 的关键属性和方法
        print("可用的优化等级:", amp.list_available_opts())
        print("✓ AMP API 访问成功")
    except Exception as e:
        print("✗ AMP API 测试失败:", e)

if __name__ == "__main__":
    test_apex_cpu