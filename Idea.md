

An embodied grasping method with spatial state awareness

**1.Introduction**

目前，LLM已经在各个方面取得了飞速的进展，融合了视觉模态的大型视觉语言模型LVLMs**[GPT4V,LLaVa,Qwen-2.5-VL,QWQ]**通过视觉图像与自然语言信息融合，对世界认识的泛化方面与对图像的推理能力有了巨大提升。将LVLMS引入机器人领域，使机器人获得认识世界并与世界交互的能力，具备以上能力的智能体叫做具身智能，目前具身智能领域正在快速发展,目前主要的方法有两类：

第一类方法为 **end-to-end policies **端到端策略，也叫**VLA(Vision-Language-Action)**模型，**[RT-1]**使用encoder-decoder类似架构从头训练，将机器人状态和视觉观测作为latent condition，然后用action query-baseed Transformer decoder解码出动作；**[RT-2,OpenVLA]**使用预训练的LLM/VLM，将action当成token直接预测，借鉴已经比较成熟的语言模型；**[DiffusionPolicy]**使用Diffusion Model多步降噪生成运动轨迹；**[Octo]**使用LLM压缩多模态表征，Diffusion作为action expert精细化输出action trajectories。此类方法目前泛化性较差，且需要大量训练数据。

第二类方法为**modular hierarchical policies**模块化分层规划：首先利用VLMs对世界进行**对齐**，利用向量的方法生成对世界的**限制**，然后使用LLMs生成**规划**或者**结构化语言**，最后使用底层原语实现机器人运动；此类方法具有一定的可解释性，规划的路径比较细致，且直接利用LLM无需对其额外训练，其中虽然使用了VLMs，但是其并未充分利用3D信息，并且其响应速度较慢，无法及时对世界环境做出反应。

我们受到Statler以及自动驾驶领域DriveVLM工作的启发，得到了一种思路:直接利用高理解能力的LVLMs实现对世界状态的感知，对世界状态不断更新，同时不断读取感知内容根据感知生成规划，最后实现路径的规划。具体来说，直接使用推理能力和检测能力强大的Qwen-2.5-VL得到物体框然后使用SAM2进行对齐Grounding并进行mask跟踪，利用生成的mask图，得到目前世界中具有哪些物体，在该规划中哪些物体应该远离，哪些物体需要靠近，生成json文件作为世界状态并保存，同时在读取状态中根据世界状态生成价值图，最后利用实时规划生成一系列不断更新的轨迹。



**1.首先使用Qwen-VL得到开放世界物体目标框，使用yoloe使用视觉提示进行实时目标检测**

**2.路径实时规划，选取距离最近的点为下一个点，避免线程之间的冲突导致机械臂路径回退**

**3.使用json储存开放世界目标状态**

**4.将抓取和移动规划分开，抓取规划使用anygrasp，移动规划使用Qwen-VL进行规划；得到目标点之后路径的选择使用价值图实现**



本文的主要贡献主要有：



**2.Relatedwork**

**Grounding：**Copa利用SoM和GraspNet相结合得到目标抓取点，然后利用LLM生成基于向量的限制，随后利用求解器，生成抓取位姿。Qwen-2.5-VL具有非常优越的json输出能力，将VLMs、LLMs和机器人进行结合，使机器人具备对世界的感知能力并且可以与现实世界进行积极的物理交互完成更加广泛的任务

**Interactive bridge：Voxposer**利用**Code as plicies**使用LLM以代码的形式生成规划并递归运行。

**Generate limits：Voxposer**利用3D体素图的形式生成限制。CoPa利用LVLMs生成向量之间的限制。



**3.Method**

3.1 问题描述</br>

3.2 世界状态对齐</br>

3.3建立3D价值体素图</br>

3.4高效路径规划</br>



**4.Experiments**

5.1 实验设置</br>

5.2 对真实世界操作</br>

5.3 对比实验</br>

5.4 有助于解决目前太空复杂场景下的碎片捕获问题</br>

**5.Conclusion**

在本文的工作中，文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本文本。







已经存在的工作使用3D点云生成抓取姿态**[GraspNet,CoPa]**，但是仍然没有广泛运用到机器人领域真实世界的推理与对齐，最近，一些工作使用3D对齐的方法**[PointLLM,VLM-Grounder,3D-LLM]**实现了对点云目标的对齐与推理，这样就提出了问题：如何使用已有的对3D点云的对齐，实现真正对真实世界的任务规划。
