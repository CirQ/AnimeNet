# -*- coding: utf8 -*-

import os

import tkinter
import tkinter.ttk as ttk
from PIL import Image, ImageTk

from generator import G

hair_color_list = ['brown', 'purple', 'orange', 'green', 'pink', 'black', 'blue', 'silver', 'gold', 'red', 'aqua', 'grey', 'white']
hair_style_list = ['twintails', 'ponytail', 'short hair', 'long hair']
eyes_color_list = ['aqua', 'purple', 'grey', 'green', 'yellow', 'red', 'blue', 'black', 'brown', 'pink', 'orange']
smile = ['false', 'true']
blush = ['false', 'true']

checkpoint_list = ['continuous_checkpoint', 'concrete_checkpoint', 'discrete_checkpoint']

class Application(tkinter.Tk):
    def __init__(self):
        super().__init__()
        self.title('AnimeGAN demo')
        self.image = None
        self.model = None
        self.create_widgets()
        self.set_obj_img('blank-file.png')
        self.gen = G()

    def set_obj_img(self, path, size=(256, 256)):
        self.path_img = path
        self.object_img = Image.open(self.path_img).convert('RGBA').resize(size, Image.BICUBIC)
        self.image = ImageTk.PhotoImage(self.object_img)
        self.label_img.configure(image=self.image)

    def create_widgets(self):
        # left frame ======================================================================
        self.frame_left = tkinter.LabelFrame(self, text='left frame')
        self.frame_left.pack(side=tkinter.LEFT, fill=tkinter.BOTH)

        # path in frame
        self.frame_file_name = tkinter.Frame(self.frame_left)
        self.frame_file_name.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        self.label_file_name = tkinter.Label(self.frame_file_name, text='generated iamge')
        self.label_file_name.pack(fill=tkinter.BOTH)

        # image in frame
        self.frame_img = tkinter.Frame(self.frame_left)
        self.label_img = tkinter.Label(self.frame_img, image=self.image)
        self.label_img.pack(fill=tkinter.BOTH)
        self.frame_img.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        # =================================================================================
        
        # right frame =====================================================================
        self.frame_right = tkinter.Frame(self)
        self.frame_right.pack(side=tkinter.RIGHT, fill=tkinter.BOTH)

        # radio button frame
        self.frame_radio_button = tkinter.LabelFrame(self.frame_right, text='character features')
        self.frame_radio_button.pack(side=tkinter.TOP, fill=tkinter.BOTH)

        # hair color
        self.frame_hair_color = tkinter.LabelFrame(self.frame_radio_button, text='hair color', relief='sunken')
        self.int_hair_color = tkinter.IntVar()
        for i, hair_color in enumerate(hair_color_list):
            rbtn = tkinter.Radiobutton(self.frame_hair_color, text=hair_color, foreground=hair_color, value=i, variable=self.int_hair_color)
            rbtn.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.frame_hair_color.pack(side=tkinter.TOP, fill=tkinter.BOTH)

        # hair style
        self.frame_hair_style = tkinter.LabelFrame(self.frame_radio_button, text='hair style', relief='sunken')
        self.int_hair_style = tkinter.IntVar()
        for i, hair_style in enumerate(hair_style_list):
            rbtn = tkinter.Radiobutton(self.frame_hair_style, text=hair_style, value=i, variable=self.int_hair_style)
            rbtn.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.frame_hair_style.pack(side=tkinter.TOP, fill=tkinter.BOTH)

        # eyes color
        self.frame_eyes_color = tkinter.LabelFrame(self.frame_radio_button, text='eyes color', relief='sunken')
        self.int_eyes_color = tkinter.IntVar()
        for i, eyes_color in enumerate(eyes_color_list):
            rbtn = tkinter.Radiobutton(self.frame_eyes_color, text=eyes_color, foreground=eyes_color, value=i , variable=self.int_eyes_color)
            rbtn.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.frame_eyes_color.pack(side=tkinter.TOP, fill=tkinter.BOTH)

        # smile
        self.frame_smile = tkinter.LabelFrame(self.frame_radio_button, text='smile', relief='sunken')
        self.bool_smile = tkinter.BooleanVar()
        for i, flag_smile in enumerate(smile):
            rbtn = tkinter.Radiobutton(self.frame_smile, text=flag_smile, value=i, variable=self.bool_smile)
            rbtn.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.frame_smile.pack(side=tkinter.TOP, fill=tkinter.BOTH)

        # blush
        self.frame_blush = tkinter.LabelFrame(self.frame_radio_button, text='blush', relief='sunken')
        self.bool_blush = tkinter.BooleanVar()
        for i, flag_blush in enumerate(blush):
            rbtn = tkinter.Radiobutton(self.frame_blush, text=flag_blush, value=i, variable=self.bool_blush)
            rbtn.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
        self.frame_blush.pack(side=tkinter.TOP, fill=tkinter.BOTH)

        # swith models
        self.str_model = tkinter.StringVar()
        self.model_combo = ttk.Combobox(self.frame_right, text='model', textvariable=self.str_model)
        cps = []
        for cl in checkpoint_list:
            for name in sorted(os.listdir(cl)):
                full_name = cl[:-10] + name[11:-4]
                cps.append(full_name)
        self.model_combo['values'] = cps
        self.model_combo.current(10)
        self.model_combo.pack(side=tkinter.TOP, fill=tkinter.BOTH)

        # confirm button
        self.frame_confirm = tkinter.LabelFrame(self.frame_right)
        self.frame_confirm .pack(side=tkinter.BOTTOM, fill=tkinter.BOTH)
        self.button = tkinter.Button(self.frame_confirm, text='generate', command=self.click_generate)
        self.button.pack(fill=tkinter.BOTH)
        # ===================================================================================

    def click_generate(self):
        choose_model = self.model_combo.get()
        if choose_model != self.model:
            self.model = choose_model
            strategy, _, epoch = self.model.split('_')
            dirname = strategy + '_checkpoint'
            modelname = 'checkpoint_epoch_' + epoch + '.pth'
            modelpath = os.path.join(dirname, modelname)
            self.gen.switch(modelpath)

        hair_color = self.int_hair_color.get()
        hair_style = self.int_hair_style.get()
        eyes_color = self.int_eyes_color.get()
        is_smile = self.bool_smile.get()
        is_blush = self.bool_blush.get()

        tags = [0 for _ in range(30)]
        tags[hair_color] = 1
        tags[len(hair_color_list) + hair_style] = 1
        tags[len(hair_color_list)+len(hair_style_list) + eyes_color] = 1
        tags[-2] = int(is_smile)
        tags[-1] = int(is_blush)

        dst_path = self.gen(tags)
        self.set_obj_img(dst_path)


if __name__ == '__main__':
    app = Application()
    app.mainloop()
