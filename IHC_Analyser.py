# -*- coding: utf-8 -*-
'''
Created on Wed Oct 16 17:20:28 2024

@author: Jiang Yiyang
'''

# ENVIRONMENT
import os
import numpy as np
import pandas as pd
import pyvips
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from tkinter.messagebox import showinfo
from pandastable import Table
from PIL import Image, ImageTk
from random import sample
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks_cwt

# GUI DESIGN
# Main Window
class MainWindow(tk.Tk):
    def __init__(self):
        # window parameter
        super().__init__()
        self.geometry('512x688')
        self.resizable(False, False)
        self.title('免疫组化 svs 批量处理')

        # elements
        self.var_dir_in = tk.StringVar()
        self.var_dir_in.set('无')
        self.lbl_in = tk.Label(self, text='导入位置：', anchor=tk.W)
        self.lbl_dir_in = tk.Label(self, textvariable=self.var_dir_in, anchor=tk.W)
        self.btn_dir_in = tk.Button(self, text='选择文件夹', command=self.sel_dir_in)

        self.var_dir_out = tk.StringVar()
        self.var_dir_out.set('无')
        self.lbl_out = tk.Label(self, text='导出位置：', anchor=tk.W)
        self.lbl_dir_out = tk.Label(self, textvariable=self.var_dir_out, anchor=tk.W)
        self.btn_dir_out = tk.Button(self, text='选择文件夹', command=self.sel_dir_out)

        self.lbl_downsample = tk.Label(self, text='下采样倍数：', anchor=tk.W)
        self.ent_downsample = ttk.Entry(self)
        self.ent_downsample.insert(0, '10')

        self.total = 0
        self.current = 0
        self.var_pro = tk.StringVar()
        self.var_pro.set('共 {} 个，当前第 {} 个'.format(self.total, self.current))
        self.bar_pro = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate')
        self.lbl_pro = tk.Label(self, textvariable=self.var_pro)

        self.btn_start = tk.Button(self, text='开始处理', command=self.start)

        self.lbl_undone = tk.Label(self, text='未完成列表', anchor=tk.W)
        self.lbl_done = tk.Label(self, text='已完成列表', anchor=tk.W)

        self.sc_undone_x = tk.Scrollbar(self, orient=tkinter.HORIZONTAL)
        self.sc_undone_y = tk.Scrollbar(self)
        self.lst_undone = tk.Listbox(self, xscrollcommand=self.sc_undone_x.set, yscrollcommand=self.sc_undone_y.set)
        self.sc_undone_x.config(command=self.lst_undone.xview)
        self.sc_undone_y.config(command=self.lst_undone.yview)
        self.sc_done_x = tk.Scrollbar(self, orient=tkinter.HORIZONTAL)
        self.sc_done_y = tk.Scrollbar(self)
        self.lst_done = tk.Listbox(self, xscrollcommand=self.sc_done_x.set, yscrollcommand=self.sc_done_y.set)
        self.sc_done_x.config(command=self.lst_done.xview)
        self.sc_done_y.config(command=self.lst_done.yview)

        # pack
        self.lbl_in.place(x=8, y=4, width=72, height=32)
        self.lbl_dir_in.place(x=88, y=4, width=352, height=32)
        self.btn_dir_in.place(x=440, y=4, width=64, height=32)

        self.lbl_out.place(x=8, y=40, width=72, height=32)
        self.lbl_dir_out.place(x=88, y=40, width=352, height=32)
        self.btn_dir_out.place(x=440, y=40, width=64, height=32)

        self.lbl_downsample.place(x=8, y=76, width=72, height=32)
        self.ent_downsample.place(x=88, y=76, width=64, height=32)

        self.bar_pro.place(x=8, y=112, width=496, height=32)
        self.lbl_pro.place(x=8, y=148, width=496, height=32)

        self.btn_start.place(x=440, y=148, width=64, height=32)

        self.lbl_undone.place(x=8, y=184, width=244, height=32)
        self.lbl_done.place(x=260, y=184, width=244, height=32)

        self.lst_undone.place(x=8, y=212, width=228, height=444)
        self.sc_undone_y.place(x=236, y=212, width=16, height=444)
        self.sc_undone_x.place(x=8, y=656, width=228,height=16)
        self.lst_done.place(x=260, y=212, width=228, height=444)
        self.sc_done_y.place(x=488, y=212, width=16, height=444)
        self.sc_done_x.place(x=260, y=656, width=228,height=16)

        def close_window():
            os._exit(0)

        self.protocol("WM_DELETE_WINDOW", close_window)

    def sel_dir_in(self):
        self.current = 0
        self.bar_pro['value'] = self.current
        self.update()
        self.lst_undone.delete(0, tk.END)
        self.lst_done.delete(0, tk.END)

        self.dir_in = tkinter.filedialog.askdirectory()
        self.var_dir_in.set(self.dir_in)
        files = os.listdir(self.dir_in)
        self.svs = []
        for file in files:
            if file.endswith('.svs'):
                self.svs.append(os.path.join(self.dir_in, file))
                self.lst_undone.insert(tk.END, file)
        self.total = len(self.svs)
        self.bar_pro['maximum'] = self.total
        self.var_pro.set('共 {} 个，当前第 {} 个'.format(self.total, self.current))

    def sel_dir_out(self):
        self.dir_out = tkinter.filedialog.askdirectory()
        self.var_dir_out.set(self.dir_out)

    def background(self):
        # Background
        # [Haem matrix, DAB matrix, Zero matrix]
        MODx = [0.684, 0.250, 0.0]
        MODy = [0.696, 0.500, 0.0]
        MODz = [0.183, 0.850, 0.0]

        cosx = [0.0, 0.0, 0.0]
        cosy = [0.0, 0.0, 0.0]
        cosz = [0.0, 0.0, 0.0]

        length = [0, 0, 0]
        q = [0.0 for b in range(9)]
        rLUT = [0.0 for b in range(256)]
        gLUT = [0.0 for b in range(256)]
        bLUT = [0.0 for b in range(256)]

        # flagx = False
        # flagy = False
        # flagz = False

        for d in range(3):
            cosx[d] = cosy[d] = cosz[d] = 0.0
            length[d] = np.sqrt(MODx[d]**2 + MODy[d]**2 + MODz[d]**2)
            if length[d] != 0.0:
                cosx[d] = MODx[d]/length[d]
                cosy[d] = MODy[d]/length[d]
                cosz[d] = MODz[d]/length[d]

        if cosx[1] == 0.0:
            if cosy[1] == 0.0:
                if cosz[1] == 0.0:
                    cosx[1] == cosz[0]
                    cosy[1] == cosx[0]
                    cosz[1] == cosy[0]

        if cosx[2] == 0.0:
            if cosy[2] == 0.0:
                if cosz[2] == 0.0:
                    if cosx[0]**2 + cosx[1]**2 > 1:
                        # flagx = True
                        cosx[2] = 0.0
                    else:
                        cosx[2] = np.sqrt(1.0 - cosx[0]**2 - cosx[1]**2)

                    if cosy[0]**2 + cosy[1]**2 > 1:
                        # flagy = True
                        cosy[2] = 0.0
                    else:
                        cosy[2] = np.sqrt(1.0 - cosy[0]**2 - cosy[1]**2)

                    if cosz[0]**2 + cosz[1]**2 > 1:
                        # flagz = True
                        cosz[2] = 0.0
                    else:
                        cosz[2] = np.sqrt(1.0 - cosz[0]**2 - cosz[1]**2)

        leng = np.sqrt(cosx[2]**2 + cosy[2]**2 + cosz[2]**2)

        cosx[2] = cosx[2] / leng
        cosy[2] = cosy[2] / leng
        cosz[2] = cosz[2] / leng

        for d in range(3):
            if cosx[d] == 0.0: cosx[d] = 0.001
            if cosy[d] == 0.0: cosy[d] = 0.001
            if cosz[d] == 0.0: cosz[d] = 0.001

        A = cosy[1] - cosx[1] * cosy[0] / cosx[0]
        V = cosz[1] - cosx[1] * cosz[0] / cosx[0]
        C = cosz[2] - cosy[2] * V / A + cosx[2] * (V / A * cosy[0] / cosx[0] - cosz[0] / cosx[0])

        q[2] = (-cosx[2] / cosx[0] - cosx[2] / A * cosx[1] / cosx[0] * cosy[0] / cosx[0] + cosy[2] / A * cosx[1] / cosx[0]) / C
        q[1] = -q[2] * V / A - cosx[1] / (cosx[0] * A)
        q[0] = 1.0 / cosx[0] - q[1] * cosy[0] / cosx[0] - q[2] * cosz[0] / cosx[0]
        q[5] = (-cosy[2] / A + cosx[2] / A * cosy[0] / cosx[0]) / C
        q[4] = -q[5] * V / A + 1.0 / A
        q[3] = -q[4] * cosy[0] / cosx[0] - q[5] * cosz[0] / cosx[0]
        q[8] = 1.0 / C
        q[7] = -q[8] * V / A
        q[6] = -q[7] * cosy[0] / cosx[0] - q[8] * cosz[0] / cosx[0]

        for d in range(3):
            for b in range(256):
                rLUT[255-b] = (255.0-float(b)*cosx[d])
                gLUT[255-b] = (255.0-float(b)*cosy[d])
                bLUT[255-b] = (255.0-float(b)*cosz[d])

        lut = np.array([
            rLUT,
            gLUT,
            bLUT
        ]).T

        lut = np.floor(lut)

        LUT = pyvips.Image.new_from_array(lut).bandfold().cast('uchar')

        return(q, LUT)

    def color_separation(self, tmp_in, q, LUT):
        tmp_out = np.array([[[0, 0, 0] for b in range(1024)] for c in range(1024)])                    

        R = tmp_in[:, :, 0]
        G = tmp_in[:, :, 1]
        B = tmp_in[:, :, 2]

        rLog = 255.0 * np.log(255/(R+1))/np.log(255.0)
        gLog = 255.0 * np.log(255/(G+1))/np.log(255.0)
        bLog = 255.0 * np.log(255/(B+1))/np.log(255.0)

        for d in range(3):
            rScaled = rLog * q[d*3]
            gScaled = gLog * q[d*3 + 1]
            bScaled = bLog * q[d*3 + 2]
            output = np.exp(-((rScaled + gScaled + bScaled) - 255.0) * np.log(255.0) / 255.0)
            output[output > 255] = 255
            tmp_out[:, :, d] = np.floor(output+0.5)

        tmp_out_dab = pyvips.Image.new_from_array(tmp_out[:, :, 1])
        tmp_out_dab = tmp_out_dab.maplut(LUT[1])
        tmp_out_dab.write_to_file('dab.jpg')

        tmp_out_h =  pyvips.Image.new_from_array(tmp_out[:, :, 0])
        tmp_out_h = tmp_out_h.maplut(LUT[0])
        tmp_out_h.write_to_file('h.jpg')

        # tmp.write_to_file('merge.jpg')

        return(tmp_out_dab, tmp_out_h)

    def threshold_choose(self, mode):
        q, LUT = self.background()
        title = {"dab":"选择 DAB 阈值", "h":"选择苏木素阈值"}
        index = 0
        pics = sample(self.svs, min(len(self.svs), 4))
        def change():
            nonlocal index, pics, patch, patch_in, patch_dab, patch_h
            if index < len(pics)-1:
                index += 1
                pic = pics[index]
                img = pyvips.Image.new_from_file(pic)
                patch = img.crop(img.width//2-512, img.height//2-512, 1024, 1024)
                patch_in = patch.numpy()
                patch_dab, patch_h = self.color_separation(patch_in, q, LUT)

                if mode == "dab":
                    fig = patch.numpy()
                    fig2 = np.dstack([patch_dab, patch_dab, patch_dab])
                    fig3 = Image.fromarray(fig)
                    content = ImageTk.PhotoImage(fig3)
                    lbl.configure(image=content)
                    lbl.image = content
                elif mode == "h":
                    fig = patch.numpy()
                    fig2 = np.dstack([patch_h, patch_h, patch_h])
                    fig3 = Image.fromarray(fig)
                    content = ImageTk.PhotoImage(fig3)
                    lbl.configure(image=content)
                    lbl.image = content
            else:
                self.toplevel.destroy()
                self.gthreshold = np.floor(np.mean(thres_list))
                # print("thres_list: ", thres_list)
                # print("threshold: ", self.gthreshold)
                return(self.gthreshold)


        def confirm():
            thres = float(var_thres.get())
            thres_list.append(thres)
            change()
            # toplevel.destroy()

        def func_scbar(*args):
            nonlocal patch, patch_dab, patch_h
            var_thres.set(int(np.floor(float(''.join([*args[1]]))*256)))
            scbar.set(float(var_thres.get())/256, (float(var_thres.get())+16)/256)

            thres = float(var_thres.get())
            if mode == "dab":
                fig = patch.numpy()
                fig2 = np.dstack([patch_dab, patch_dab, patch_dab])
                fig[fig2[:,:,2] < thres] = [255, 0, 0]
            elif mode == "h":
                fig = patch.numpy()
                fig2 = np.dstack([patch_h, patch_h, patch_h])
                fig[fig2[:,:,2] < thres] = [0, 0, 255]
            fig3 = Image.fromarray(fig)
            content = ImageTk.PhotoImage(fig3)
            lbl.configure(image=content)
            lbl.image = content

        def func_ent(*args):
            nonlocal patch, patch_dab, patch_h
            scbar.set(float(var_thres.get())/256, (float(var_thres.get())+16)/256)

            thres = float(var_thres.get())
            if mode == "dab":
                fig = patch.numpy()
                fig2 = np.dstack([patch_dab, patch_dab, patch_dab])
                fig[fig2[:,:,2] < thres] = [255, 0, 0]
            elif mode == "h":
                fig = patch.numpy()
                fig2 = np.dstack([patch_h, patch_h, patch_h])
                fig[fig2[:,:,2] < thres] = [0, 0, 255]
            fig3 = Image.fromarray(fig)
            content = ImageTk.PhotoImage(fig3)
            lbl.configure(image=content)
            lbl.image = content

        self.toplevel = tk.Toplevel(self)
        lbl = tk.Label(self.toplevel)
        scbar = tk.Scrollbar(self.toplevel, command=func_scbar, orient=tk.HORIZONTAL)
        lbl_thres = tk.Label(self.toplevel, text="阈值: ")
        var_thres = tk.StringVar()
        var_thres.set("0")
        ent_thres = tk.Entry(self.toplevel, textvariable=var_thres)
        btn = tk.Button(self.toplevel, text="确认阈值", command=lambda:confirm())

        ent_thres.bind("<KeyRelease>", func_ent)

        self.toplevel.title(title[mode])
        self.toplevel.geometry("512x592")
        self.toplevel.resizable(False, False)
        lbl.place(x=0, y=0, width=512, height=512)
        scbar.place(x=8, y=520, width=368,height=32)
        lbl_thres.place(x=376, y=520, width=64, height=32)
        ent_thres.place(x=444, y=520, width=64, height=32)
        btn.place(x=192, y=560, width=128, height=32)

        thres_list = []

        pic = pics[index]
        img = pyvips.Image.new_from_file(pic)
        patch = img.crop(img.width//2-512, img.height//2-512, 1024, 1024)
        patch_in = patch.numpy()
        patch_dab, patch_h = self.color_separation(patch_in, q, LUT)

        if mode == "dab":
            fig = patch.numpy()
            fig2 = np.dstack([patch_dab, patch_dab, patch_dab])
            fig3 = Image.fromarray(fig)
            content = ImageTk.PhotoImage(fig3)
            lbl.configure(image=content)
            lbl.image = content
        elif mode == "h":
            fig = patch.numpy()
            fig2 = np.dstack([patch_h, patch_h, patch_h])
            fig3 = Image.fromarray(fig)
            content = ImageTk.PhotoImage(fig3)
            lbl.configure(image=content)
            lbl.image = content

    def threshold(self, tmp, threshold):
        arr = tmp.numpy().flatten()
        tmp_thres = arr[arr < threshold]

        return(tmp_thres)

    def mean_od (self, tmp_thres):
        area = len(tmp_thres)/(1024*1024)
        grey = tmp_thres.astype(np.float64)
        grey[grey == 0] = 0.5
        grey[grey == 255] = 254.5
        od = 0.434294481 * np.log(255/grey)

        mod = np.mean(od)

        return(mod, area)
        pass

    def statistics(self, svs, dataframe):
        df_h_filtered = dataframe[(dataframe["mod_h"] >= dataframe["mod_h"].quantile(0.05)) & (dataframe["area_h"] >= dataframe["area_h"].quantile(0.05))]
        name = os.path.basename(svs)
        mod = df_h_filtered["mod_dab"].mean()
        area = df_h_filtered["area_dab"].mean()

        res = pd.DataFrame({"name":[name], "mod":[mod], "area":[area]})
        return(res)


    def show_result(self, result):
        top = tk.Toplevel(self)
        frame = tk.Frame(top)
        frame.pack()
        table = Table(frame, dataframe=result, showtoolbar=True, showstatusbar=True)
        table.show()
        top.mainloop()

    def start(self):
        self.gthreshold = 0
        self.current = 0

        files = os.listdir(self.dir_in)
        self.svs = []
        for file in files:
            if file.endswith('.svs'):
                self.svs.append(os.path.join(self.dir_in, file))
                self.lst_undone.insert(tk.END, file)
        self.total = len(self.svs)
        self.bar_pro['maximum'] = self.total
        self.var_pro.set('共 {} 个，当前第 {} 个'.format(self.total, self.current))
        self.update()

        downsample = int(self.ent_downsample.get())
        q, LUT = self.background()

        self.threshold_choose("dab")
        self.wait_window(self.toplevel)
        threshold_dab = self.gthreshold
        # print(threshold_dab)
        self.threshold_choose("h")
        self.wait_window(self.toplevel)
        threshold_h = self.gthreshold
        # print(threshold_h)

        # Main Loop
        result = pd.DataFrame(columns=["name", "mod", "area"])
        for svs in self.svs:
            # svs = r'D:\_Bussiness\脑膜瘤组化\2024-07-28\B202310357-1-100-NDUFA4L2.svs'
            img = pyvips.Image.new_from_file(svs)
            dataframe = pd.DataFrame(columns=["name", "mod_dab", "area_dab", "mod_h", "area_h"])
            dataframe["name"]  = dataframe["name"].astype("object")
            dataframe["mod_dab"]  = dataframe["mod_dab"].astype("float64")
            dataframe["area_dab"]  = dataframe["area_dab"].astype("float64")
            dataframe["mod_h"]  = dataframe["mod_h"].astype("float64")
            dataframe["area_h"]  = dataframe["area_h"].astype("float64")

            for i in range(img.width // (1024 * downsample)):
                for j in range(img.height // (1024 * downsample)):
            # for i in range(30, 31):
                # for j in range(31, 32):
                    tmp = img.crop(1024*downsample*i, 1024*downsample*j, 1024, 1024)
                    tmp_in = tmp.numpy()
                    # tmp = img.crop(1024*i, 1024*j, 1024, 1024)
                    # tmp.write_to_file(os.path.join(self.dir_out, os.path.basename(svs)) + '-{}-{}.tiff'.format(i, j))

                    tmp_dab, tmp_h = self.color_separation(tmp_in, q, LUT)

                    tmp_dab_thres = self.threshold(tmp_dab, threshold_dab)
                    mod_dab, area_dab = self.mean_od(tmp_dab_thres)

                    tmp_h_thres = self.threshold(tmp_h, threshold_h)
                    mod_h, area_h = self.mean_od(tmp_h_thres)

                    data = pd.DataFrame({"name":[svs], "mod_dab":[mod_dab], "area_dab":[area_dab], "mod_h":[mod_h], "area_h":[area_h]})
                    dataframe = pd.concat([dataframe, data]).reset_index(drop=True)
                    # print(dataframe)

            if not(dataframe.empty):
                result = pd.concat([result, self.statistics(svs, dataframe)]).reset_index(drop=True)
            else:
                # print("empty")
                showinfo("警告", "下采样倍数过大")

            for t in range(len(self.lst_undone.get(0, tk.END))):
                if os.path.basename(svs) == self.lst_undone.get(0, tk.END)[t]:
                    self.lst_undone.delete(t)
                    self.lst_done.insert(tk.END, os.path.basename(svs))
                    break

            self.current += 1
            self.var_pro.set('共 {} 个，当前第 {} 个'.format(self.total, self.current))
            self.bar_pro['value'] = self.current
            self.update()

        try:
            result.to_csv(os.path.join(self.dir_out, "result.csv"))
        except:
            showinfo("错误", "未选择存储位置，请在结果窗口弹出后务必手动存储")    

        self.show_result(result)


if __name__ == '__main__':
    main = MainWindow()
    main.mainloop()
