# 钢铁、铜、铝、塑料、聚碳酸酯、木头、玻璃、陶瓷、混凝土、石头(大理石)
# 密度，杨氏模量，泊松比，alpha, beta（瑞利阻尼系数） 
class MatSet():
    Steel = 7850, 2.0E11, 0.29, 20, 3e-8
    Copper = 8960, 1.2E11, 0.35, 5, 1e-7 # zeta=0.01
    Aluminum = 2700, 6.9E10, 0.33, 5, 1.5e-7 # zeta=0.01
    Plastic = 1070, 1.4E9, 0.35, 20, 3e-7#1E-6
    Polycarbonate = 1190, 2.4E9, 0.37, 0.5, 4E-7
    Wood = 750, 1.1E10, 0.25, 60, 2E-6
    Glass = 2600, 6.2E10, 0.20, 1, 1E-7
    Ceramic = 2700, 7.2E10, 0.19, 6, 1E-7
    Concrete = 2400, 3.0E10, 0.2, 10, 8E-7 # zeta=0.05
    Stone = 2700, 5.0E10, 0.25, 25, 1.5E-6 # zeta=0.1
    
    # 用中括号索引的方式获取材料属性
    def __getitem__(self, index):
        return list(self.__class__.__dict__.values())[index + 1] 
    