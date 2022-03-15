# yolov5-based-CSGO-Bot
I trained the yolov5 model using my own labelled data, and developed the complete pipeline.  

My goal is not sharing hacks, so no labelled data or trained models will be shared here. And I don't use this in multi-player games. 

( 
This is the initial version I developed last year. And since this is only motivated by my personal interest, so everything was done in rush. <br>
The code could be cleaned and redesigned in the future when I have time. 
) 

---
# Demo


The bounding boxes are draw on the image. Green means teammates, and red means enemies. 

When there is an enemy in the screen, the program will move mouse cursor until the enemy is at the center of screen, and then the program will perform the click action to fire. 

Enemy heads and bodys are being detected seperately, then grouped into people.

When a head is detected, the program will aim onto the head; otherwise, it will aim onto enemy body.

<a id="bot" href="https://github.com/hanmmmmm/yolov5-based-CSGO-Bot/blob/main/gifs/csgo_bot_4.gif">
    <img src="https://github.com/hanmmmmm/yolov5-based-CSGO-Bot/blob/main/gifs/csgo_bot_4.gif" alt="bot gif" title="bot" width="700"/>
</a>
<br><br>

<a id="bot" href="https://github.com/hanmmmmm/yolov5-based-CSGO-Bot/blob/main/gifs/csgo_bot_2.gif">
    <img src="https://github.com/hanmmmmm/yolov5-based-CSGO-Bot/blob/main/gifs/csgo_bot_2.gif" alt="bot gif" title="bot" width="700"/>
</a>


--- 
# Label
I labelled all the trainning data myself, using Yolo-mark and makesense for labelling. 

<a id="label" href="https://github.com/hanmmmmm/yolov5-based-CSGO-Bot/blob/main/gifs/csgo_label_3.gif">
    <img src="https://github.com/hanmmmmm/yolov5-based-CSGO-Bot/blob/main/gifs/csgo_label_3.gif" alt="label gif" title="label" width="550"/>
</a>








