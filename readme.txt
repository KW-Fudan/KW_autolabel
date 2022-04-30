1、首先将待标注的图片放到unlabel目录下，标注生成的结果将放在datasets/cvoid/labels/train/（存放坐标信息）和datasets/cvoid/images/train/(存放标注图片)两个目录下

2、执行autopred.py文件，会弹出标注窗口：
a)鼠标左键拖动可设置矩形线框标注抗原盒子的区域
b)如果对标注不满意，鼠标右键在点击已标注的矩形线框可以删除矩形线框重新标注。
c）标注完成后英文输入法按y确认，按n跳过（即放弃，跳下一张），也可以用左右方向键选择待标注图片（此处注意英文输入法，否则会报错）

注：文件首次启动时需要下载 https://ultralytics.com/assets/Arial.ttf文件，放入C:\Users\xxx\AppData\Roaming\Ultralytics目录下，如果自动下载失败可以手动下载并放入该目录
