
from ast import copy_location
from operator import le
import sys
import cv2
from pprint import pprint
import numpy as np
import mss
import math
from numpy.lib.function_base import select
import psutil

import pyautogui
# from torch._C import T
# from torch._C import uint8
# from torch._C import uint8
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

import time
import random

from utils.yolo5_detect import yolo5_detector
from utils.getwin import get_active_window
from utils.bbox_iou import compute_iou
from utils.get_gpu_state import get_gpu_state


class my_csgo_aimbot():
    def __init__(self) -> None:

        self.FLAG_use_detection = True

        self.img = None
        self.sct = mss.mss()

        if self.FLAG_use_detection:

            model_path = 'yolo_models/v2/exp2/weights/best.pt'
            self.infe_image_size = 640
            self.detector = yolo5_detector( model_path, self.infe_image_size )

            self.names = ['T','T_head','T_die','C','C_head',\
            'C_die','tm_ind','em_ind','purchase' , 'ending_logo' , 'healthBar', 'scoreBoard' , 'healthBar', 'scoreBoard' ]
            
            self.names = [ 'P','H','D','P','H','D','P','H','D','TM', 'HP', 'PW', 'END', 'SC' ]
            
            self.class_index = {}
            
            for i in range(len(self.names)):
                self.class_index[self.names[i]] = i
                
            self.yolo_result = None



        self.num_of_obj = 0
        self.detected_obj = []
        
        self.em_num = 0
        
        self.people_bboxs = []
        self.head_bboxs = []
        self.tm_bboxs = []
        
        self.detection_final_result = None
        
        
        self.all_bbox = None
        
        self.obj_exist = False
        
        self.loop_duration = 0
        self.yolo_inference_time = 0
        self.yolo_inference_post_time = 0
        self.get_screen_duration = 0


        self.last_check_scene_time = 0

        ###  screen parameters
        self.game_window_x = 0
        self.game_window_y = 90
        self.game_window_w = 1280
        self.game_window_h = 960

        self.screen_box = {"top": self.game_window_y-30, "left": self.game_window_x, "width": self.game_window_w, "height": self.game_window_h, "mon": -1}

        self.game_center_in_screen = ( int(self.game_window_w/2 + self.game_window_x) , int(self.game_window_h/2 + self.game_window_y)  ) # x , y 
        
        self.target_xy_to_center = [0,0]
        
        self.current_target_bbox_info = []
        
        ### visulization 
        self.show_size = (640, 480)

        self.UI = np.ones((870,640,3)).astype(np.uint8)
        self.UI_base_color = (120,120,120)
        self.UI[:,:,:] = self.UI_base_color
        # self.UI *= 190

        self.img_TL_in_UI = (0,0)  # row , col
        self.img_BR_in_UI = ( self.img_TL_in_UI[0] + self.show_size[1] , self.img_TL_in_UI[1] + self.show_size[0] )
        
    
        ### gpu info 
        self.gpuID = 0
        self.gpu_name = ''
        self.gpu_temp = 0
        self.gpu_load = 0
        self.gpu_mem_total = 0
        self.gpu_mem_used = 0
        
        self.cpu_usage_history_length = 60
        self.cpu_usage_history = np.zeros(( self.cpu_usage_history_length ))
        self.cpu_usage_history_ct = 0
        
        ### action 
        self.current_scene = 0  # 0:death/menu/loading, 1:play, 2:purchase, 3:ending  
        self.scene_names = ['death/menu/loading', 'play', 'purchase', 'ending']
        self.allow_click = False


    def get_screen(self):
        t1 = time.time()
        self.img = np.array( self.sct.grab(self.screen_box) )
        self.img = cv2.cvtColor( self.img, cv2.COLOR_RGBA2RGB )
        t2 = time.time()
        self.get_screen_duration = int( (t2-t1)*1000.0 )
        # print('get screen : {} ms'.format(  int( (t2-t1)*1000.0 ) ) )



    def detect(self):
        self.detected_obj = []
        
        t1 = time.time()
        p = self.detector.predict( self.img )[0]
        
        t2 = time.time()
        yolo_time = int( (t2-t1)*1000.0 )
        self.yolo_inference_time = yolo_time
        # print('detection time {} ms, fps: {:.1f}'.format( yolo_time, 1000/yolo_time ))
        
        self.yolo_result = p.cpu().detach().numpy()

        self.num_of_obj = self.yolo_result.shape[0]

        # make up a zero-value element if it is empty, so it won't cause error later 
        if self.num_of_obj == 0:
            self.yolo_result = np.zeros((1,6)).astype(np.int)
            self.yolo_result[0,5] = -1
            self.obj_exist = False
        else:
            self.obj_exist = True
            
        
        # convert it into integer type
        self.yolo_result[:,4] *= 100
        self.yolo_result = self.yolo_result.astype(np.int32)
        
        if self.obj_exist:
            self.detected_obj = self.yolo_result[:,5].tolist()
            self.detected_obj.sort()
            
            for i in range(len(self.detected_obj)):
                self.detected_obj[i] = self.names[self.detected_obj[i]]
            
        t3 = time.time()
        self.yolo_inference_post_time = int( (t3-t2) * 1000 )
        
        # print('people bboxs:')
        # print(self.people_bboxs)
        # print('head bboxs:')
        # print(self.head_bboxs)
        # print('teammate-name bboxs:')
        # print(self.tm_bboxs)

        
    
    def bbox_class_split(self):
        self.people_bboxs = []
        self.head_bboxs = []
        self.tm_bboxs = []
        
        p = self.yolo_result.copy()
        for i in range( p.shape[0] ):
            class_id = p[i][-1]  
            if class_id in [ 0,3,6 ]:
                self.people_bboxs.append(p[i,0:5])
            elif class_id in [ 1,4,7 ]:
                self.head_bboxs.append(p[i,0:5])
            elif class_id in [ 9 ]:
                self.tm_bboxs.append(p[i,0:5])
                
        self.people_bboxs = np.array( self.people_bboxs )
        self.head_bboxs   = np.array( self.head_bboxs )
        self.tm_bboxs     = np.array( self.tm_bboxs )
        
        
        
    # def remove_duplicate_bbox(self, bboxs):
    
    #     iou_thresh = 0.8
        
    #     loop_count = 0
    #     while loop_count != bboxs.shape[0]:
    #         cand = bboxs[loop_count]
    #         considered = []
    #         for i in range( loop_count+1, bboxs.shape[0] ):
    #             cand2 = bboxs[i]
    #             cand_convert = ( cand[1], cand[0], cand[3], cand[2] )
    #             cand2_convert = ( cand2[1], cand2[0], cand2[3], cand2[2] )
    #             iou_score = compute_iou(cand_convert, cand2_convert)
    #             if iou_score >= iou_thresh:
    #                 considered.append(i)
    #         if len(considered) != 0:
                
                    

    #     cleaned_bbox = []
        
    #     return cleaned_bbox
        
    
    
        
    def teammate_enemy_split(self):
        
        # print('grouping detected objs')
        
        # first, convert the x0 y0 x1 y1 format into cx cy w h 
        heads = np.zeros( ( self.head_bboxs.shape[0] , 4) )  # center_x, center_y, w, h
        for i in range( heads.shape[0]):
            # print(i)
            heads[i][0] = int( (self.head_bboxs[i][0] + self.head_bboxs[i][2] ) / 2 )
            heads[i][1] = int( (self.head_bboxs[i][1] + self.head_bboxs[i][3] ) / 2 )
            heads[i][2] = int(  self.head_bboxs[i][2] - self.head_bboxs[i][0] )
            heads[i][3] = int(  self.head_bboxs[i][3] - self.head_bboxs[i][1] )
        
        body = np.zeros( ( self.people_bboxs.shape[0] , 4) ) # center_x, center_y, w, h
        for i in range( body.shape[0]):
            # print(i)
            body[i][0]  = int( (self.people_bboxs[i][0] + self.people_bboxs[i][2] ) / 2 )
            body[i][1]  = int( (self.people_bboxs[i][1] + self.people_bboxs[i][3] ) / 2 )
            body[i][2]  = int(  self.people_bboxs[i][2] - self.people_bboxs[i][0] )
            body[i][3]  = int(  self.people_bboxs[i][3] - self.people_bboxs[i][1] )
            
        tms = np.zeros( ( self.tm_bboxs.shape[0] , 4) ) # center_x, center_y, w, h
        for i in range( tms.shape[0]):
            # print(i)
            tms[i][0]   = int( (self.tm_bboxs[i][0] + self.tm_bboxs[i][2] ) / 2 )
            tms[i][1]   = int( (self.tm_bboxs[i][1] + self.tm_bboxs[i][3] ) / 2 )
            tms[i][2]   = int(  self.tm_bboxs[i][2] - self.tm_bboxs[i][0] )
            tms[i][3]   = int(  self.tm_bboxs[i][3] - self.tm_bboxs[i][1] )
        
        # print('\nAfter converting ')
        # print( 'heads : \n{}\n'.format(heads) )
        # print( 'body  : \n{}\n'.format(body) )
        
        # print(self.img.shape)
        
        # second, merge the bodys and heads, into target-points
        pp = []  # [ [ [ class, x , y ] ,[], [], ...  ],  class: 0 -> body&head, 1 -> head, 2 -> body
        for i in range( heads.shape[0] ):
            
            the_class = -1 # assumption
            this_head_finish = False 
            for b in range( body.shape[0] ):
                # print('for {}-th head, {}-th body'.format(i, b))
                # print( 'heads : \n{}\n'.format(heads) )
                # print( 'body  : \n{}\n'.format(body) )
                if ( abs( heads[i][0] - body[b][0]) < heads[i][2] * 1.0 ) and ( body[b][1]-( body[b][3]/2 ) < heads[i][1] < body[b][1] ) :
                    this_head_finish = True
                    pp.append([ 0 , heads[i][0], heads[i][1], heads[i][2], heads[i][3] ])
                    body = np.delete( body, b ,0 )
                    break
                    
                    
            if this_head_finish == False:
                pp.append( [ 1 , heads[i][0], heads[i][1], heads[i][2], heads[i][3]  ] ) 
                this_head_finish = True    
                
        if len(body) != 0:  # some body-bbox with no head-bbox 
            for j in range( len(body) ):
                pp.append( [ 2, body[j][0], body[j][1], body[j][2], body[j][3]  ] )
                
        
            
        
        
        # third, check each element in pp, if it is tm or em 
        for i in range(len(pp)):
            
            if pp[i][0] == 0 or pp[i][0] == 1: # when the target area is a head
                # if pp[i][4] > 20:
                #     y_range_to_check = [ pp[i][2] - pp[i][4] * 1.6, pp[i][2] - pp[i][4] * 0.5 ]
                #     x_range_to_check = [ pp[i][1] - pp[i][3] * 1.0, pp[i][1] + pp[i][3] * 1.0 ]
                # else: 
                #     y_range_to_check = [ pp[i][2] - pp[i][4] * 3.6, pp[i][2] - pp[i][4] * 0.5 ]
                #     x_range_to_check = [ pp[i][1] - pp[i][3] * 2.0, pp[i][1] + pp[i][3] * 2.0 ]
                
                y_low = pp[i][2] - pp[i][4] * 0.5
                y_h = 13 + pp[i][4] * 0.6
                y_top = y_low - y_h
                
                y_range_to_check = [ y_top, y_low ]
                x_range_to_check = [ pp[i][1] - pp[i][3] * 2.0, pp[i][1] + pp[i][3] * 2.0 ]
                    
            if pp[i][0] == 2: # when the target area is a body
                y_range_to_check = [ pp[i][2] - pp[i][4] * 0.9, pp[i][2] - pp[i][4] * 0.3 ]
                x_range_to_check = [ pp[i][1] - pp[i][3] * 1.0, pp[i][1] + pp[i][3] * 1.0 ]
                
            start_point = (  int(x_range_to_check[0]*2) , int(y_range_to_check[0]*2)   )
            end_point   = (  int(x_range_to_check[1]*2) , int(y_range_to_check[1]*2)   )
            
            # print('tm searching in:')
            # print( pp[i], start_point, end_point)

            if pp[i][0] == 0 or pp[i][0] == 1:
                tm_searching_color = (255,255,255)
            else:
                tm_searching_color = (0,0,0)
            self.img = cv2.rectangle( self.img, start_point, end_point, tm_searching_color , 3)
        
            found_tm = False
            for t in range(len(tms)):
                if x_range_to_check[0] <= tms[t][0] <= x_range_to_check[1]:
                    if y_range_to_check[0] <= tms[t][1] <= y_range_to_check[1]:
                        pp[i].append( 0 )  # teammate
                        found_tm = True 
                        break
                    
            if found_tm == False:
                pp[i].append( 1 ) # enemy 
                
        

        for p in range(len(pp)):
            # if pp[p][0] == 0:
            #     color = (255,0,0)
            # if pp[p][0] == 1:
            #     color = (0,255,0)
            # if pp[p][0] == 2:
            #     color = (0,0,255)
            if pp[p][-1] == 0:
                color = (0,255,0)
            if pp[p][-1] == 1:
                color = (0,0,255)
            self.img = cv2.circle(self.img, (int(pp[p][1]*2) , int(pp[p][2]*2)) , 30, color=color, thickness=5)
        
        
        # return pp
        self.detection_final_result = np.array( pp ).astype(np.int) 
        
        # pprint(self.detection_final_result)
    
    

        
            
    def check_scene(self):
        # t1 = time.time()
        # check the current scene
        detected_classes = self.yolo_result[:,-1]
        
        if self.class_index['PW'] in detected_classes:
            self.current_scene = 2
        
        elif self.class_index['END'] in detected_classes:
            self.current_scene = 3
        
        elif self.class_index['HP'] in detected_classes:
            self.current_scene = 1
        
        else:
            self.current_scene = 0
            
        # t2 = time.time()    
        
        # check if mouse click is allowed 
        if self.current_scene == 1:
            # tbf = time.time()
            current_window = get_active_window()
            # taf = time.time()
            if 'Counter-Strike' in current_window :
                self.allow_click = True
            else:
                self.allow_click = False
        else:
            self.allow_click = False
            
        # t3 = time.time()
        # print('check scene {}  {} '.format( int((t2-t1)*1000),int((taf-tbf)*1000) )  )    
            
            
            
    def dim_image(self, img, dimValue):
        img = (img*dimValue).astype(np.uint8)
        return img
            
    def make_mouse_command(self):
        
        corrected_target_xy = self.detection_final_result.copy()
        original_leng = corrected_target_xy.shape[0]
        tm_indices = []
        # print('target before remove tm')
        # print(corrected_target_xy)
        for i in range(corrected_target_xy.shape[0]):
            if corrected_target_xy[i,-1] == 0:
                tm_indices.append(i)
        corrected_target_xy = np.delete(corrected_target_xy, tm_indices ,0)
        # print('tm indixes:',tm_indices)
        # print('target before remove tm')
        # print(corrected_target_xy)
        
        self.em_num = original_leng - len(tm_indices)
        
        if self.em_num == 0:
            self.target_xy_to_center = [ 0,0 ]
            return
        
        corrected_target_xy[:,1:5] *= 2
        corrected_target_xy[:,1] += self.game_window_x
        corrected_target_xy[:,2] += self.game_window_y
        
        pix_distance_to_center = []
        
        for i in range(corrected_target_xy.shape[0]):
            xy   = np.array( [corrected_target_xy[i, 1],     corrected_target_xy[i,2] ])
            cxcy = np.array( [self.game_center_in_screen[0], self.game_center_in_screen[1]])
            # dist = math.sqrt( np.linalg.norm(xy - cxcy) )
            dist = int(  np.linalg.norm(xy - cxcy) ) 
            pix_distance_to_center.append(dist)
            
        if len(pix_distance_to_center) > 0:
            closest_obj_i = pix_distance_to_center.index( min(pix_distance_to_center)  )   
        else:
            closest_obj_i = 0
            
        xy   = np.array( [corrected_target_xy[closest_obj_i, 1],     corrected_target_xy[closest_obj_i,2] ])
        self.target_xy_to_center = [ xy[0]-cxcy[0], xy[1]-cxcy[1] ]
        # print('pix to move ',self.target_xy_to_center)
        self.current_target_bbox_info = corrected_target_xy[closest_obj_i]
        # print('c: {}   closest i: {}'.format( self.game_center_in_screen, closest_obj_i ))
        # print(pix_distance_to_center)
        
        self.img = cv2.line(self.img , (int(self.game_window_w/2),0) , (int(self.game_window_w/2), self.game_window_h) , (0,255,0), 2)
        self.img = cv2.line(self.img , (0, int(self.game_window_h/2)) , (self.game_window_w, int(self.game_window_h/2) ) , (0,255,0), 2)
    
    
    
    def attack(self):
        if self.allow_click and self.em_num != 0:
            
            # target bbox info 
            w    = self.current_target_bbox_info[3]
            h    = self.current_target_bbox_info[4]
            type = self.current_target_bbox_info[5]
            
            if type == 0 or type == 1:
                mouse_speed_ratio = 1.0
                check_reached_critiria = (w+h)/2 
            elif type == 2:
                mouse_speed_ratio = 1.0
                check_reached_critiria = (w+h)/2 
            
            self.target_xy_to_center[0] = int( self.target_xy_to_center[0] * mouse_speed_ratio )
            self.target_xy_to_center[1] = int( self.target_xy_to_center[1] * mouse_speed_ratio )
            # print('moving : {}'.format( self.target_xy_to_center ))
            pyautogui.move( self.target_xy_to_center[0] , self.target_xy_to_center[1] )
            if abs(self.target_xy_to_center[0]) < check_reached_critiria :
                if abs(self.target_xy_to_center[0]) < check_reached_critiria:
                    pyautogui.click()
        else:
            pass
            # print('not moving mouse ')
        

    def draw_bbox(self):
        for i in range(self.yolo_result.shape[0]):
            
            bbox = self.yolo_result[i,0:4]
            bbox[0] = int( bbox[0] * self.game_window_w / self.infe_image_size ) 
            bbox[1] = int( bbox[1] * self.game_window_w / self.infe_image_size )
            bbox[2] = int( bbox[2] * self.game_window_w / self.infe_image_size ) 
            bbox[3] = int( bbox[3] * self.game_window_w / self.infe_image_size )

            start_point = ( bbox[0], bbox[1]  )
            end_point =   ( bbox[2], bbox[3]  )

            class_id = int(self.yolo_result [i,5])
            
            # print('class_id: {}'.format(class_id))
            obj_name = self.names[class_id]

            font = cv2.FONT_HERSHEY_SIMPLEX
            textorg = ( bbox[0], bbox[1]-10 )
            fontScale = 2

            if class_id in [0,3,6 ]:
                color = (255, 0, 0)
            elif class_id in [1,4,7 ]:
                color = (255, 0, 0)
            elif class_id in [9 ]:
                color = (0, 255, 0)
            else:
                color = (255, 255, 100)
                
                
            thickness = 2
            # self.img = cv2.putText(self.img, obj_name , textorg, font, fontScale, color, thickness, cv2.LINE_AA)

            bbox_line_thickness = 3
            self.img = cv2.rectangle( self.img, start_point, end_point, color, bbox_line_thickness)


    def put_info(self):
        
        tbp = time.time()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        
        # clear the text area in self.UI 
        self.UI[480:, :] = self.UI_base_color
        
        scene_string = 'Current scene: ' + self.scene_names[self.current_scene]
        color = (255,255,255)
        self.UI = cv2.putText(self.UI, scene_string , (10, 500), font, fontScale, color, thickness, cv2.LINE_AA)
        
        bbox_string = 'Find bboxs: {} person  {} head  {} tmName'.format(self.people_bboxs.shape[0], self.head_bboxs.shape[0], self.tm_bboxs.shape[0]) 
        self.UI = cv2.putText(self.UI, bbox_string , (10, 530), font, fontScale, color, thickness, cv2.LINE_AA)

        total_num_p = app.detection_final_result.shape[0]
        if total_num_p > 0:
            total_num_em = int( np.sum( app.detection_final_result[:,-1] ) )
            total_num_tm = int( total_num_p - total_num_em  )
        elif total_num_p == 0:
            total_num_em = 0
            total_num_tm = 0
        em_tm_string = 'Find:   {} enemy   {} teammate'.format(total_num_em, total_num_tm)
        self.UI = cv2.putText(self.UI, em_tm_string , (10, 560), font, fontScale, color, thickness, cv2.LINE_AA)

        allow_attack_string = 'Allow attack: {}'.format(self.allow_click)
        self.UI = cv2.putText(self.UI, allow_attack_string , (10, 590), font, fontScale, color, thickness, cv2.LINE_AA)

        pix_move_string = 'Moving cursor: x {}  y {}'.format( self.target_xy_to_center[0], self.target_xy_to_center[1])
        self.UI = cv2.putText(self.UI, pix_move_string , (10, 620), font, fontScale, color, thickness, cv2.LINE_AA)


        gpu_string = 'GPU: Load {}%  Mem {}/{}MB  Temp {}C'.format(self.gpu_load, self.gpu_mem_used, self.gpu_mem_total, self.gpu_temp)
        self.UI = cv2.putText(self.UI, gpu_string , (10, 650), font, fontScale, color, thickness, cv2.LINE_AA)

        cpu_usage = str(int(psutil.cpu_percent()))
        self.cpu_usage_history[self.cpu_usage_history_ct] = cpu_usage
        self.cpu_usage_history_ct += 1
        if self.cpu_usage_history_ct >= self.cpu_usage_history_length:
            self.cpu_usage_history_ct = 0
        cpu_usage_avg = str( int( np.mean(self.cpu_usage_history) ) )
        
        if len(cpu_usage_avg) == 1:
            cpu_usage_avg = ' '+cpu_usage_avg
        cpu_string = 'CPU: Load {}%  Mem {}/{}MB '.format( cpu_usage_avg, int( psutil.virtual_memory()[3]/(1024*1024)) , int( psutil.virtual_memory()[0]/(1024*1024)) )
        self.UI = cv2.putText(self.UI, cpu_string , (10, 680), font, fontScale, color, thickness, cv2.LINE_AA)

        time_string = 'Time milli-seconds:'
        self.UI = cv2.putText(self.UI, time_string , (10, 790), font, fontScale, color, thickness, cv2.LINE_AA)
        
        tap = time.time()
        time_for_add_these_info = int((tap - tbp)*1000)
        # print('add text info, time: {}'.format( (tap - tbp)*1000 ))
        
        time_string = 'Main loop {}  get screen {}  detect {}  '.format(self.loop_duration, self.get_screen_duration, self.yolo_inference_time)
        self.UI = cv2.putText(self.UI, time_string , (10, 820), font, fontScale, color, thickness, cv2.LINE_AA)
        time_string = 'Add these info {}  '.format(time_for_add_these_info)
        self.UI = cv2.putText(self.UI, time_string , (10, 850), font, fontScale, color, thickness, cv2.LINE_AA)



    def showIt(self):
        self.img = cv2.resize(self.img , self.show_size )
        self.UI[self.img_TL_in_UI[0]:self.img_BR_in_UI[0], self.img_TL_in_UI[1]:self.img_BR_in_UI[1] ] = self.img

        cv2.imshow('detect', self.UI  )
        cv2.waitKey(1)


        
    def update_gpu_state(self):
        # gpustate = get_gpu_state()
        self.gpuID, self.gpu_name, self.gpu_load, self.gpu_temp, self.gpu_mem_total, self.gpu_mem_used = get_gpu_state()
        
        
# update loop timestamp, and print in terminal         
def loop_timer(lasttime):
    nowTime = time.time()
    dt = int( ( nowTime - lasttime )  * 1000.0 )
    #print('time from last loop: {} ms'.format(dt))
    return nowTime , dt       



if __name__=="__main__":
    
    app = my_csgo_aimbot()
    # app.check_scene()

    lasttime = time.time()
    
    last_gpu_time = time.time()

    while True:
        
        lasttime, app.loop_duration = loop_timer(lasttime) 
        
        if time.time() - last_gpu_time > 1.0:
            app.update_gpu_state()
            last_gpu_time = time.time()
        
        app.get_screen() 
        
        app.detect() 
        
        t1 = time.time()
        app.img = app.dim_image(app.img, 0.8)
        t2 = time.time()
        if time.time() - app.last_check_scene_time > 2.0:
            app.check_scene()
            app.last_check_scene_time = time.time()
        t3 = time.time()
        app.bbox_class_split()
        t4 = time.time()
        app.teammate_enemy_split()
        t5 = time.time()
        app.draw_bbox()
        t6 = time.time()
        app.make_mouse_command()
        t7 = time.time()
        app.attack()
        t8 = time.time()
        
        
        # tbp = time.time()
        app.put_info()
        t9 = time.time()
        # tap = time.time()
        # print('add text info, time: {}'.format( (tap - tbp)*1000 ))
        
        app.showIt() 
        t10 = time.time()
        
        print( app.yolo_inference_post_time , int((t2-t1)*1000) , int((t3-t2)*1000) , int((t4-t3)*1000)\
             , int((t5-t4)*1000) , int((t6-t5)*1000) , int((t7-t6)*1000) , int((t8-t7)*1000),\
                  int((t9-t8)*1000) , int((t10-t9)*1000))
        
        






