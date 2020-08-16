# Automatic-Traffic-Control-system
    
To run the code in terminal, use following commands

For Phase 1:
python3 main.py -y yolo_pretrained -i phase1.mp4 -o phase1_final.avi -n 2 -p 1

For Phase 2:
python3 main.py -y yolo_pretrained -i phase2.mp4 -o phase2_final.avi -n 2 -p 2

For Phase 3:
python3 main.py -y yolo_pretrained -i phase3.mp4 -o phase3_final.avi -n 2 -p 3

For Phase 4:
python3 main.py -y yolo_pretrained -i phase4.mp4 -o phase4_final.avi -n 2 -p 4

Note: yolo_pretrained is a directory containing pre-trained YOLO weights
    - It should have three files:
    	- <name_of_file>.names
    	- <name_of_file>.cfg
    	- <name_of_file>.weights
    

For testing use smaller video input_video3.mp4, by using following command:
python3 main.py -y yolo-coco -i input_video3.mp4 -o out3.avi -n 3