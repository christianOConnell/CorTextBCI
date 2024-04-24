import csv

def readEEG(filename, dataCube):
    # dataCube should contain marker and timestamp data
    with open(filename, 'r', newline='') as eeg:
        temp2 = ['f',dataCube[-1][1] + 4]
        dataCube.append(temp2)
        temp = []
        curr = dataCube[0]
        nextBoi = dataCube[1]
        startTime = float(curr[1]) #curr time
        nextInd = 2

        sums = [0,0,0,0,0,0,0,0] # 8 0s
        count = 0 # for finding averages

        eegreader = csv.reader(eeg, delimiter=',')
        next(eegreader) # skip headers on first line
        for row in eeg:

            splitRow = row.strip().split(',')
            #print("row",splitRow)
            if curr[0] == 'l' or curr[0] == -1:
                curr[0] = -1  # -1 for left
            else:
                curr[0] = 1  # 1 for right
            #print(float(splitRow[8]), "", startTime, "", float(splitRow[8]), "",float(nextBoi[1]))
            if( float(splitRow[8]) < float(nextBoi[1])):
                if(float(splitRow[8]) > startTime + 0.25):
                    #temp.append(splitRow[:-1]) #don't append timestamp
                    for i in range(len(splitRow) - 1):
                        sums[i] += float(splitRow[i])
                    count+=1
                    if count >= 9:
                        for i in range(len(sums)):
                            sums[i] = int(sums[i])//10
                            temp.append(sums[i])
                        dataCube[nextInd - 2].append(temp)
                        #print("temp",temp)
                        temp = []
                        sums = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 0s
                        count = 0

            else:
                curr = nextBoi
                if nextInd >= len(dataCube):
                    dataCube[nextInd - 2].pop(1)
                    dataCube[nextInd - 2].append(temp)
                    dataCube.pop(-1)
                    #print("break")
                    break
                nextBoi = dataCube[nextInd]
                dataCube[nextInd - 2].pop(1)
                dataCube[nextInd - 2].append(temp)
                nextInd += 1
                temp = []
                startTime = float(curr[1])

    return dataCube #first col is markers, 2nd col is channel 0 data, 3rd, col is channel 1 data, etc.
                                        # 3D axis is data samples in the channel




def readMarkers(filename):
    with open(filename, 'r', newline='') as markers:
        dataCube = []
        eegreader = csv.reader(markers, delimiter=',')
        next(eegreader)
        firstRow = next(eegreader)
        firstTime = float(firstRow[1])
        dataCube.append([firstRow[0], 0])
        for row in markers:
            splitRow = row.strip().split(',')
            arr = [splitRow[0], round(float(splitRow[1]) - firstTime)] # mark timestamps start at 0
            dataCube.append(arr)
    return dataCube


def main():
    markerCSV = "test3.csv"
    eegCSV = "eegData1.csv"
    theDataCube = readMarkers(markerCSV)
    #print(theDataCube)
    theDataCube = readEEG(eegCSV, theDataCube)
    #print(theDataCube)
    for i in range(len(theDataCube)):
        for j in range(0,len(theDataCube[i])):
            if j == 0:
                print(theDataCube[i][0])
            else :
                for k in range(len(theDataCube[i][j])):
                    print(theDataCube[i][j][k], end=' ')
            print()


main()
