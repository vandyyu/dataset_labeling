# encoding: utf-8
#TODO : 优化文本展示的位置
"""
嗯。。OCR相关标注
"""
__author__ = "yuwanli"
import tkinter as tk
from tkinter import *
import os
import cv2
import time
import copy
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil

font_path = os.path.join(os.getcwd(), "simhei.ttf")
ENCODING = "utf-8"


def shrink(img, boxes, rate=0.6):
    img = cv2.resize(img, None, fx=rate, fy=rate)
    boxes = [np.multiply(box, rate).astype(np.int32) for box in boxes]
    return img, boxes


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def drawChineseText(cvImage, xy, text, color=(255, 0, 0), font_size=20, tff=font_path):
    pilImage = Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilImage)
    draw.text(
        xy,
        text,
        color,
        ImageFont.truetype(tff, font_size, encoding=ENCODING)
    )
    return cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)


def get_boxes_texts_from_file(fpath):
    """
    fpath文件格式和ICDAR公开数据集的标注格式一致
    [x1, x2, x3, ... , xn, text], 点按照顺时针标注
    如果框内没有文本，就用###代替
    :param fpath:
    :return:
    """
    boxes = []
    texts = []
    if fpath is None:
        return boxes, texts
    with open(fpath, encoding=ENCODING) as rf:
        for i, line in enumerate(rf):
            if line.startswith("\ufeff"):
                line = line[1:]
            items = line.split(",")
            # 合法的标记框，split后一定是奇数个
            if len(items) % 2 != 1:
                continue
            try:
                box = np.array(items[:-1], dtype=np.float32)
            except:
                print("line {} failed! {} -> {}".format(i, fpath, str(items[:-1])))
                continue
            box = np.reshape(box, (-1, 2))
            boxes += [box]
            texts += [items[-1].strip()]
    return boxes, texts


class LabelAdjuster(object):
    nst_thresh = 12  # 小于此距离时，自动黏合
    cbox_rect_min_size = (10, 10)  # 创建矩形box的最小尺寸，小于该尺寸则不存储
    cbox_polygan_min_area = 400  # 创建多边形box的最小外接圆面积

    def __init__(self, image, boxes, texts, win_name, scaling=0.6):
        """
        :param image: 待矫正的图片矩阵，opencv格式读取的BGR格式，会自动根据指定的scaling进行缩放，
                        在缩放后的图片上进行标注矫正。但是在标注结束后，会将缩放的坐标进行还原，因此
                        最终返回的矫正位置依然是相对于原图像尺寸下的标注坐标。
        :param boxes: 对image进行预先标注的标注框坐标信息，例如两个标注框：
                        [[[10, 20], [50, 20], [50, 50], [10, 50]],
                        [[100, 200], [500, 200], [500, 500], [100, 500]]]，同样地，会根据scaling进行同步缩放
        :param texts:  每个标注框对应的文本内容。与boxes中的每个框一一对应。例如：['文本一', '文本二']
        :param win_name: 当前标注窗口界面的标题
        :param scaling: 标注界面的缩放比例
        """
        assert isinstance(image, np.ndarray), "请输入图片矩阵！"
        assert isinstance(boxes, (np.ndarray, list)), "请输入boxes标注矩阵或者list！"
        assert isinstance(texts, (np.ndarray, list)), "请输入texts标注矩阵或者list！"
        assert scaling > 0

        # boxes的最外层一定要是list，list里面存放多个不同尺寸的box
        # （虽然最外层也可以是ndarray，但是这样会导致存放的所有box都必须是同一个尺寸，否则会报错）
        if isinstance(boxes, np.ndarray):
            boxes = list(boxes)
        if isinstance(texts, np.ndarray):
            texts = list(texts)
        image, boxes = shrink(image, np.asarray(boxes), scaling)
        self.win_name = win_name
        self.scaling = scaling

        self.cur_text_i = np.Inf  # 当前编辑的是第几个文本框
        # self.cur_box_previous_text = None  # 当前文本框在更新之前的文本内容
        self.cur_box_text = "###"
        self.boxes = boxes
        self.texts = texts
        self.image = image
        self.original_image = copy.deepcopy(self.image)  # 没有任何绘制的原始图片
        self.init_base_face()

        # cur_box_i, cur_box_j，而是boxes索引的位置, boxes[cur_box_i, cur_box_j]才表是一个点
        self.cur_box_i, self.cur_box_j = np.Inf, np.Inf
        self.cur_key = -1

        # 用鼠标右键创造一个box的x和y的值,(x1, y1)是左上角，(x2, y2)是右下角
        self.cbox_x1 = np.Inf
        self.cbox_y1 = np.Inf
        self.cbox_x2 = np.Inf
        self.cbox_y2 = np.Inf
        self.cbox_startX, self.cbox_startY = np.Inf, np.Inf
        self.cbox_rect_valid = True  # 创建的box是否是有效的
        self.cur_del_box_index = -1  # 记录当前删除的box的索引
        self.rbtn_timestamp = -1  # 右键按下去时的时间戳
        self.rclk_mode = False  # 标志右键点击开始
        self.cbox_polygon_buf = []  # 缓存当前绘制的多边形点
        self.cbox_polygon_stop = False  # 停止绘制多边形的标志

        # 当右键双击，进入内容标注模式
        self.text_modify_mode = False
        self.text_modify_mode_locked = False  # 同一个样本，一次只能修改一个box，同步

    def init_base_face(self):
        """
        最初始的界面中，有box框和文本
        :return:
        """
        self.image = copy.deepcopy(self.original_image)
        self.image = self._draw_texts()
        self.image_cp = copy.deepcopy(self.image)
        self.image = cv2.drawContours(self.image, np.array(self.boxes), -1, (0, 255, 0), 2)

    def get_nearest_box_index(self, x, y):
        """
        获取距离p点最近的box的位置
        :param p, 目标点
        :param boxes, 图片中所有的boxes
        :return: boxes中的索引号
        """
        nst_i = np.Inf
        nst_j = np.Inf
        nst_value = np.Inf
        for i, box in enumerate(self.boxes):
            for j, p_box in enumerate(box):
                dst = distance(p_box, [x, y])
                if dst < nst_value and dst < self.nst_thresh:
                    nst_value = dst
                    nst_i = i
                    nst_j = j
        return nst_i, nst_j

    def get_box_index(self, x, y):
        """
        获取(x, y)在哪个box上
        :param x:
        :param y:
        :return:
        """
        for i, box in enumerate(self.boxes):
            if cv2.pointPolygonTest(box, (x, y), False) > 0:
                return i
        return -1

    def update_box_pos(self, x, y):
        """矫正box的位置，把(x,y)附近的点移动到(x, y)"""
        self.image = copy.deepcopy(self.image_cp)  # 方法触发时刷新
        cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
        if self.cur_box_i != np.Inf and self.cur_box_i != self.cur_del_box_index:
            self.boxes[self.cur_box_i][self.cur_box_j] = [x, y]

    def create_rect_box(self, x, y):
        self.cbox_x1 = min(self.cbox_startX, x)
        self.cbox_y1 = min(self.cbox_startY, y)
        self.cbox_x2 = max(self.cbox_startX, x)
        self.cbox_y2 = max(self.cbox_startY, y)
        if self.cbox_x1 == np.Inf or \
                        self.cbox_x2 == np.Inf or \
                        self.cbox_y1 == np.Inf or \
                        self.cbox_y2 == np.Inf:
            return False
        self.image = copy.deepcopy(self.image_cp)  # 方法触发时刷新
        cv2.rectangle(self.image, (self.cbox_x1, self.cbox_y1), (self.cbox_x2, self.cbox_y2), (0, 0, 255), 2)
        # 太小的box不要
        if (self.cbox_x2 - self.cbox_x1) <= self.cbox_rect_min_size[0] or (self.cbox_y2 - self.cbox_y1) <= \
                self.cbox_rect_min_size[1]:
            return False
        return True

    def create_polygon_box(self, x, y):
        """多边形创建结束则返回True"""
        success = False
        self.image = copy.deepcopy(self.image_cp)  # 方法触发时刷新
        if len(self.cbox_polygon_buf) > 1:
            for i in range(1, len(self.cbox_polygon_buf)):
                cv2.line(self.image, tuple(self.cbox_polygon_buf[i - 1]), tuple(self.cbox_polygon_buf[i]), (0, 0, 255),
                         2)

        # 当绘制最后一个点的时候，要特殊处理
        if len(self.cbox_polygon_buf) > 1 and distance([x, y], self.cbox_polygon_buf[0]) <= self.nst_thresh:
            # 至少有三个点才能构成多边形，此时才能结束
            if len(self.cbox_polygon_buf) >= 3:
                success = True
                cv2.line(self.image, tuple(self.cbox_polygon_buf[-1]), tuple(self.cbox_polygon_buf[0]), (0, 0, 255), 2)
        else:
            cv2.line(self.image, tuple(self.cbox_polygon_buf[-1]), (x, y), (0, 0, 255), 2)
        return success

    def _add_box(self, box):
        self.boxes = list(self.boxes)
        self.texts = list(self.texts)
        box = np.array(box)
        center, radius = cv2.minEnclosingCircle(box)
        if np.pi * radius * radius > self.cbox_polygan_min_area:
            self.boxes.append(box)
            # 同时加入一个新的空文本
            self.texts.append("###")
        return self.boxes

    def _clear_cbox_polygon(self):
        """将创建多边形box操作恢复到初始化"""
        self.cbox_polygon_buf = []
        self.cbox_polygon_stop = False
        self.rclk_mode = False

    def handle_boxes_event(self, event, x, y, flags, params):
        """
        文本信息标注、生成标注框、矫正标注框
        :param event:
        :param x:
        :param y:
        :param flags:
        :param params:
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cur_box_i, self.cur_box_j = self.get_nearest_box_index(x, y)
            self._clear_cbox_polygon()
            pass

        if event == cv2.EVENT_RBUTTONDOWN:
            # 右键按下时，记录时间戳，同时标志着即将进行多边形绘制，所以记录坐标
            self.cbox_startX, self.cbox_startY = x, y
            self.rbtn_timestamp = time.time()

        if event == cv2.EVENT_RBUTTONUP:
            # 右键抬起时，存储创建的box
            if self.cbox_rect_valid:
                self._add_box([[self.cbox_x1, self.cbox_y1],
                               [self.cbox_x2, self.cbox_y1],
                               [self.cbox_x2, self.cbox_y2],
                               [self.cbox_x1, self.cbox_y2]])
            # 右键单击事件的触发时机
            if time.time() - self.rbtn_timestamp <= 0.5 and not self.cbox_rect_valid:
                self.rclk_mode = True
                if not self.cbox_polygon_stop:
                    self.cbox_polygon_buf.append([x, y])
                else:
                    # 如果绘制结束了，再单击右键，则存储绘制的多边形，在此之前，没有存储，可以撤销边
                    self._add_box(self.cbox_polygon_buf)
                    self._clear_cbox_polygon()

        if event == cv2.EVENT_LBUTTONDBLCLK:
            # 左键双击删除box
            cur_index = self.get_box_index(x, y)
            if cur_index != -1:
                self.cur_del_box_index = cur_index
                # 删除box
                if 0 <= cur_index < len(self.boxes):
                    self.boxes = np.delete(self.boxes, cur_index, 0)
                if 0 <= cur_index < len(self.texts):
                    self.texts = np.delete(self.texts, cur_index, 0)
                self.init_base_face()
                # 重置删除状态
                self.cur_del_box_index = -1

        if flags == 1:
            # 左键滑动的时候，矫正更新
            self.update_box_pos(x, y)
        if flags == 2:
            # 右键划一条线，创建矩形box
            self.cbox_rect_valid = self.create_rect_box(x, y)

        if self.rclk_mode:
            # 右键点击之后，开始创建多边形box
            if not self.cbox_polygon_stop:
                self.cbox_polygon_stop = self.create_polygon_box(x, y)

        if event == cv2.EVENT_MBUTTONDOWN:
            # 创建多边形box的时候，点击鼠标中间按键，撤销上一次绘制
            if len(self.cbox_polygon_buf) > 1:
                del self.cbox_polygon_buf[-1]
            # 执行了撤销操作，说明还能继续
            if self.cbox_polygon_stop:
                self.cbox_polygon_stop = False

    def modify_box_text(self, event):
        self.cur_box_text = self.edit_text.get("0.0", "end").strip()
        self.edit_text_root.destroy()

    def show_edit_text(self, callback, default_text=""):
        self.edit_text_root = tk.Tk()
        self.edit_text_root.update()
        self.edit_text_root.deiconify()

        self.edit_text_root.title("纠正文本内容")
        self.edit_text_root.resizable(False, False)
        self.edit_text_root.bind('<Control-s>', callback)

        self.edit_text = tk.Text(self.edit_text_root, fg='white', bg='black', font=font_path, width=30, height=5,
                                 insertbackground="red", insertofftime=10, insertwidth=2, spacing1=10,
                                 spacing2=10, spacing3=30, padx=10)
        self.edit_text.insert("end", default_text)
        self.edit_text.grid(sticky=tk.N + tk.E + tk.W)
        self.edit_text.focus()
        self.edit_text_root.mainloop()

    def lock_edit_text(self):
        # 文本编辑模式，上锁
        self.text_modify_mode = True
        self.text_modify_mode_locked = True

    def unlock_edit_text(self):
        # 释放锁
        self.text_modify_mode_locked = False
        self.text_modify_mode = False
        self._clear_cbox_polygon()

    def update_text(self):
        if 0 <= self.cur_text_i < len(self.texts):
            self.texts[self.cur_text_i] = self.cur_box_text
            self.init_base_face()  # 避免文本出现重叠，所以必须重新初始化界面

    def onMouse(self, event, x, y, flags, params=None):
        if event == cv2.EVENT_RBUTTONDBLCLK:
            # 右键在box内双击，此时进入文本编辑模式
            self.cur_text_i = self.get_box_index(x, y)
            if self.cur_text_i != -1:
                self.text_modify_mode = True
                if 0 <= self.cur_text_i < len(self.texts):
                    self.cur_box_text = self.texts[self.cur_text_i]
                    # self.cur_box_previous_text = self.cur_box_text

        if self.text_modify_mode:
            if not self.text_modify_mode_locked:
                self.lock_edit_text()
                self.show_edit_text(self.modify_box_text, self.cur_box_text)
                self.unlock_edit_text()
                self.update_text()
        else:
            self.handle_boxes_event(event, x, y, flags, params)

    def setKeyboardMotion(self):
        if self.cur_box_i == np.Inf:
            return
        if self.cur_box_i == self.cur_del_box_index:
            return
        cur_p = self.boxes[self.cur_box_i][self.cur_box_j]
        # 上移
        if self.cur_key == 2490368:
            cur_p[1] -= 1
            cur_p[1] = max(0, cur_p[1])
            self.update_box_pos(cur_p[0], cur_p[1])
        # 下
        if self.cur_key == 2621440:
            cur_p[1] += 1
            cur_p[1] = min(self.image.shape[0], cur_p[1])
            self.update_box_pos(cur_p[0], cur_p[1])
        # 左
        if self.cur_key == 2424832:
            cur_p[0] -= 1
            cur_p[0] = max(0, cur_p[0])
            self.update_box_pos(cur_p[0], cur_p[1])
        # 右
        if self.cur_key == 2555904:
            cur_p[0] += 1
            cur_p[0] = min(self.image.shape[1], cur_p[0])
            self.update_box_pos(cur_p[0], cur_p[1])

    def draw_texts(self, image, texts, texts_xy, i):
        """
        在图片上绘制文本。为避免额外的资源消耗，只有在图片像素发生变化的时候，才会更新
        :param image:
        :param texts:  多个文本
        :param texts_xy:  多个文本对应的绘制位置
        :param i:  绘制第i个，如果等于-1则表示绘制所有
        :return:
        """
        assert -1 <= i < len(texts)
        if i == -1:
            for text, xy in zip(texts, texts_xy):
                image = drawChineseText(image, xy, text)
        else:
            image = drawChineseText(image, texts_xy[i], texts[i])
        return image

    def _draw_texts(self):
        texts_xy = [[np.min(box[:, 0]), np.min(box[:, 1])] for box in self.boxes]
        self.image = self.draw_texts(self.image, self.texts, texts_xy, -1)
        return self.image

    def make_boxes_correct(self):
        """
        对已经标注的图片，进行人工检查，矫正
        :param img, 需要矫正的图片
        :param boxes, 图片上初步打标数据
        :return: 校正后的boxes以及文本，此box返回后是相对于原始图像尺寸的坐标，
                  而在对象中的self.boxes依然保持相对于缩放后图像尺寸的坐标。
        """
        cv2.namedWindow(self.win_name, cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)
        # cv2.resizeWindow(self.win_name, 800, 600)
        cv2.setMouseCallback(self.win_name, self.onMouse)

        while self.cur_key != 32 and cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) != 0:
            # 切记一定不要这样连续的、无条件的去直接替换(刷新原图)，因为这样极快的刷新频率，
            # 会使得任何界面操作(例如画线、画矩形)都会被立马覆盖成原图，导致界面会一直不变。
            # 要在具体的动作事件中添加刷新(替换)操作。
            if self.cur_key == ord('d') or self.cur_key == 'D':
                break
            self.image = cv2.drawContours(self.image, np.array(self.boxes), -1, (0, 255, 0), 2)
            cv2.imshow(self.win_name, self.image)
            self.cur_key = cv2.waitKeyEx(10)
            self.setKeyboardMotion()
        boxes = [np.multiply(box, 1. / self.scaling).astype(np.int32) for box in self.boxes]
        return boxes, self.texts


def make_label(win_name, image_path, label_path=None, scaling=0.5, destroy_win=True):
    """
    单张图片标注。
    对LabelAdjuster类传入的参数进一步封装.
    :param win_name:
    :param image_path: 待标注图片路径
    :param label_path: 预标注，ICDAR格式一致
    :param scaling: 缩放比例
    :param destroy_win: 标注结束后是否关闭窗口
    :return: 返回boxes和texts格式和LabelAdjuster()传入的boxes和texts参数格式一致。
    """
    boxes, texts = get_boxes_texts_from_file(label_path)
    adjuster = LabelAdjuster(cv2.imread(image_path), boxes, texts, win_name, scaling)
    boxes, texts = adjuster.make_boxes_correct()
    # 按d或者D删除当前图片
    if adjuster.cur_key == ord('d') or adjuster.cur_key == ord('D'):
        os.remove(image_path)
        boxes, texts = [], []
    if destroy_win:
        cv2.destroyWindow(win_name)
    return boxes, texts


def save(save_path, boxes, texts):
    if boxes is None or texts is None:
        return
    if len(boxes) == 0 or len(texts) == 0:
        return
    assert len(boxes) == len(texts), "save {} failed! boxes array and texts array must be the same length.".format(
        save_path)
    with open(save_path, encoding=ENCODING, mode='w') as wf:
        for box, text in zip(boxes, texts):
            box = ",".join(np.array(box).flatten().astype(np.str))
            line = box + "," + text + "\n"
            wf.write(line)


def batch_labeling_generator(images_root, labels_root=None, ignores=[]):
    """
    批量图片标注样本生成器。
    :param images_root: 包含所有图片的目录
    :param labels_root: 预标记的标签目录，和ICDAR数据集一致，None表示不存在预标记
    :return: 待标注图片路径，预标记的文件路径
    """
    images_name = [name for name in os.listdir(images_root) if name.split(".")[0] not in ignores]
    if not images_name:
        return [], []
    if labels_root is None:
        for name in images_name:
            yield os.path.join(images_root, name), None
    else:
        names = []
        for name in os.listdir(labels_root):
            assert name.startswith("gt_"), "label name must be start with gt_"
            names += [name[3:].split(".")[0]]
        for name in images_name:
            if name.split(".")[0] not in names:
                yield os.path.join(images_root, name), None
            else:
                label_path = os.path.join(labels_root, "gt_{}.txt".format(name.split(".")[0]))
                yield os.path.join(images_root, name), label_path


def batch_labeling(images_root, labels_root, output_dir, scaling=0.5):
    """
    批量标注.
    如果发现标注错了，把output_dir目录下对应标注的文件删除就行了。
    :param images_root: 包含所有图片的目录
    :param output_dir: 标注输出目录
    """
    ignores = []
    if os.path.exists(output_dir):
        ignores = [name[3:].split(".")[0] for name in os.listdir(output_dir) if name.startswith("gt_")]
    else:
        os.makedirs(output_dir)
    for image_path, label_path in \
            batch_labeling_generator(images_root, labels_root, ignores):
        win_name = os.path.basename(image_path)
        boxes, texts = make_label(win_name, image_path, label_path, scaling=scaling, destroy_win=False)

        # 点击关闭窗口，退出标注
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) == 0:
            break
        cv2.destroyWindow(win_name)

        # 点击了关闭则退出，不会保存，按空格保存，并进入下一帧
        if label_path is not None:
            save_path = os.path.join(output_dir, os.path.basename(label_path))
        else:
            save_path = os.path.join(output_dir, "gt_{}.txt".format(win_name.split(".")[0]))
        save(save_path, boxes, texts)


if __name__ == '__main__':
    cur_dir = os.getcwd()
    images_root = os.path.join(cur_dir, "samples", "images")
    labels_root = os.path.join(cur_dir, "samples", "labels")  # 如果没有可以为None
    output_dir = os.path.join(cur_dir, "samples", "labels_correct")
    batch_labeling(images_root, labels_root, output_dir, scaling=0.5)
