import pyautogui as gui

def readBrain(filePath):
    brain = open(filePath)
    ms = ""
    sum = 0
    sumy = 0
    numLines = 1
    for line in brain:
        arr = line.strip().split(", ")
        if(ms == line.strip()[-1]):
            sum += float(arr[1]) * 0.001
            sumy += float(arr[2]) * 0.001
            numLines += 1

        else:
            gui.moveRel(sum//numLines, sumy//numLines, duration=0.05)
            ms = line[-1]
            sum = float(arr[1]) * 0.001
            sumy = float(arr[2]) * 0.001
            numLines = 1

def main():
    gui.moveRel(500.5,500, duration = 1)
    readBrain("OpenBCI-RAW-2023-10-31_15-13-34.txt")

main()