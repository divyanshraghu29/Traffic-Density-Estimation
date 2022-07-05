import os.path

import cv2
import numpy as np
import pandas as pd


def getOutput(videoPath):
    cap1 = cv2.VideoCapture(videoPath)

    frames_count1, fps1, width1, height1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap1.get(cv2.CAP_PROP_FPS), cap1.get(
        cv2.CAP_PROP_FRAME_WIDTH), cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width1 = int(width1)
    height1 = int(height1)
    frametime1 = 10
    # print(frames_count1, fps1, width1, height1)

    # creates a pandas data frame with the number of rows the same length as frame coun
    df = pd.DataFrame(index=range(int(frames_count1)))
    df.index.name = "Frames1"

    framenumber1 = 0  # keeps track of current frame
    carscrossedup1 = 0  # keeps track of cars that crossed up
    carscrosseddown1 = 0  # keeps track of cars that crossed down
    carids1 = []  # blank list to add car ids
    caridscrossed1 = []  # blank list to add car ids that have crossed
    totalcars1 = 0  # keeps track of total cars

    fgbg1 = cv2.createBackgroundSubtractorMOG2()
    ret, frame1 = cap1.read()  # import image
    ratio1 = .5  # resize ratio
    image1 = cv2.resize(frame1, (0, 0), None, ratio1, ratio1)  # resize image
    width21, height21, channels1 = image1.shape
    video1 = cv2.VideoWriter(os.path.dirname(videoPath) + '/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             fps1,
                             (height21, width21),
                             1)

    while True:
        ret, frame1 = cap1.read()  # import image

        if ret:  # if there is a frame continue with code

            image1 = cv2.resize(frame1, (0, 0), None, ratio1, ratio1)  # resize image

            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # converts image to gray

            fgmask1 = fgbg1.apply(gray1)  # uses the background subtraction

            # applies different thresholds to fgmask to try and isolate cars
            # just have to keep playing around with settings until cars are easily identifiable
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing1 = cv2.morphologyEx(fgmask1, cv2.MORPH_CLOSE, kernel1)
            opening1 = cv2.morphologyEx(closing1, cv2.MORPH_OPEN, kernel1)
            dilation1 = cv2.dilate(opening1, kernel1)
            retvalbin1, bins1 = cv2.threshold(dilation1, 220, 255, cv2.THRESH_BINARY)  # removes the shadows

            # creates contours
            contours1, hierarchy1 = cv2.findContours(bins1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # line created to stop counting contours, needed as cars in distance become one big contour
            lineypos1 = 20
            cv2.line(image1, (0, lineypos1), (120, lineypos1), (255, 0, 0), 2)

            # line y position created to count contours
            lineypos21 = 240
            cv2.line(image1, (0, lineypos21), (120, lineypos21), (0, 255, 0), 2)

            # min area for contours in case a bunch of small noise contours are created
            minarea = 250

            # max area for contours, can be quite large for buses
            maxarea = 50000

            # vectors for the x and y locations of contour centroids in current frame
            cxx1 = np.zeros(len(contours1))
            cyy1 = np.zeros(len(contours1))

            for i in range(len(contours1)):  # cycles through all contours in current frame

                if hierarchy1[
                    0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)

                    area1 = cv2.contourArea(contours1[i])  # area of contour

                    if minarea < area1 < maxarea:  # area threshold for contour

                        # calculating centroids of contours
                        cnt1 = contours1[i]
                        M = cv2.moments(cnt1)
                        cx1 = int(M['m10'] / M['m00'])
                        cy1 = int(M['m01'] / M['m00'])

                        if ((cy1 > 0)):  # filters out contours that are above line (y starts at top)

                            # gets bounding points of contour to create rectangle
                            # x,y is top left corner and w,h is width and height
                            x, y, w, h = cv2.boundingRect(cnt1)

                            # creates a rectangle around contour
                            cv2.rectangle(image1, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Prints centroid text in order to double check later on
                            cv2.putText(image1, str(cx1) + "," + str(cy1), (cx1 + 10, cy1 + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        .3, (0, 0, 255), 1)

                            cv2.drawMarker(image1, (cx1, cy1), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                           line_type=cv2.LINE_AA)

                            # adds centroids that passed previous criteria to centroid list
                            cxx1[i] = cx1
                            cyy1[i] = cy1

            # eliminates zero entries (centroids that were not added)
            cxx1 = cxx1[cxx1 != 0]
            cyy1 = cyy1[cyy1 != 0]

            # empty list to later check which centroid indices were added to dataframe
            minx_index21 = []
            miny_index21 = []

            # maximum allowable radius for current frame centroid to be considered the same centroid from previous frame
            maxrad1 = 25

            # The section below keeps track of the centroids and assigns them to old carids or new carids

            if len(cxx1):  # if there are centroids in the specified area

                if not carids1:  # if carids is empty

                    for i in range(len(cxx1)):  # loops through all centroids

                        carids1.append(i)  # adds a car id to the empty list carids
                        df[str(carids1[i])] = ""  # adds a column to the dataframe corresponding to a carid

                        # assigns the centroid values to the current frame (row) and carid (column)
                        df.at[int(framenumber1), str(carids1[i])] = [cxx1[i], cyy1[i]]

                        totalcars1 = carids1[i] + 1  # adds one count to total cars

                else:  # if there are already car ids

                    dx1 = np.zeros((len(cxx1), len(carids1)))  # new arrays to calculate deltas
                    dy1 = np.zeros((len(cyy1), len(carids1)))  # new arrays to calculate deltas

                    for i in range(len(cxx1)):  # loops through all centroids

                        for j in range(len(carids1)):  # loops through all recorded car ids

                            # acquires centroid from previous frame for specific carid
                            oldcxcy1 = df.iloc[int(framenumber1 - 1)][str(carids1[j])]

                            # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                            curcxcy1 = np.array([cxx1[i], cyy1[i]])

                            if not oldcxcy1:  # checks if old centroid is empty in case car leaves screen and new car shows

                                continue  # continue to next carid

                            else:  # calculate centroid deltas to compare to current frame position later

                                dx1[i, j] = oldcxcy1[0] - curcxcy1[0]
                                dy1[i, j] = oldcxcy1[1] - curcxcy1[1]

                    for j in range(len(carids1)):  # loops through all current car ids

                        sumsum1 = np.abs(dx1[:, j]) + np.abs(dy1[:, j])  # sums the deltas wrt to car ids

                        # finds which index carid had the min difference and this is true index
                        correctindextrue1 = np.argmin(np.abs(sumsum1))
                        minx_index1 = correctindextrue1
                        miny_index1 = correctindextrue1

                        # acquires delta values of the minimum deltas in order to check if it is within radius later on
                        mindx1 = dx1[minx_index1, j]
                        mindy1 = dy1[miny_index1, j]

                        if mindx1 == 0 and mindy1 == 0 and np.all(dx1[:, j] == 0) and np.all(dy1[:, j] == 0):
                            # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                            # delta could be zero if centroid didn't move

                            continue  # continue to next carid

                        else:

                            # if delta values are less than maximum radius then add that centroid to that specific carid
                            if np.abs(mindx1) < maxrad1 and np.abs(mindy1) < maxrad1:
                                # adds centroid to corresponding previously existing carid
                                df.at[int(framenumber1), str(carids1[j])] = [cxx1[minx_index1], cyy1[miny_index1]]
                                minx_index21.append(
                                    minx_index1)  # appends all the indices that were added to previous carids
                                miny_index21.append(miny_index1)

                    for i in range(len(cxx1)):  # loops through all centroids

                        # if centroid is not in the minindex list then another car needs to be added
                        if i not in minx_index21 and miny_index21:

                            df[str(totalcars1)] = ""  # create another column with total cars
                            totalcars1 = totalcars1 + 1  # adds another total car the count
                            t1 = totalcars1 - 1  # t is a placeholder to total cars
                            carids1.append(t1)  # append to list of car ids
                            df.at[int(framenumber1), str(t1)] = [cxx1[i], cyy1[i]]  # add centroid to the new car id

                        elif curcxcy1[0] and not oldcxcy1 and not minx_index21 and not miny_index21:
                            # checks if current centroid exists but previous centroid does not
                            # new car to be added in case minx_index2 is empty

                            df[str(totalcars1)] = ""  # create another column with total cars
                            totalcars1 = totalcars1 + 1  # adds another total car the count
                            t1 = totalcars1 - 1  # t is a placeholder to total cars
                            carids1.append(t1)  # append to list of car ids
                            df.at[int(framenumber1), str(t1)] = [cxx1[i], cyy1[i]]  # add centroid to the new car id

            #         # The section below labels the centroids on screen

            currentcars1 = 0  # current cars on screen
            currentcarsindex1 = []  # current cars on screen carid index

            for i in range(len(carids1)):  # loops through all carids

                if df.at[int(framenumber1), str(carids1[i])] != '':
                    # checks the current frame to see which car ids are active
                    # by checking in centroid exists on current frame for certain car id

                    currentcars1 = currentcars1 + 1  # adds another to current cars on screen
                    currentcarsindex1.append(i)  # adds car ids to current cars on screen

            for i in range(currentcars1):  # loops through all current car ids on screen

                # grabs centroid of certain carid for current frame
                curcent1 = df.iloc[int(framenumber1)][str(carids1[currentcarsindex1[i]])]

                # grabs centroid of certain carid for previous frame
                oldcent1 = df.iloc[int(framenumber1 - 1)][str(carids1[currentcarsindex1[i]])]

                if curcent1:  # if there is a current centroid

                    # On-screen text for current centroid

                    if oldcent1:  # checks if old centroid exists
                        # adds radius box from previous centroid to current centroid for visualization
                        xstart1 = oldcent1[0] - maxrad1
                        ystart1 = oldcent1[1] - maxrad1
                        xwidth1 = oldcent1[0] + maxrad1
                        yheight1 = oldcent1[1] + maxrad1
                        cv2.rectangle(image1, (int(xstart1), int(ystart1)), (int(xwidth1), int(yheight1)), (0, 125, 0),
                                      1)

                        # checks if old centroid is on or below line and curcent is on or above line
                        # to count cars and that car hasn't been counted yet
                        if oldcent1[1] >= lineypos21 and curcent1[1] <= lineypos21 and carids1[
                            currentcarsindex1[i]] not in caridscrossed1:

                            carscrossedup1 = carscrossedup1 + 1
                            cv2.line(image1, (0, lineypos21), (width1, lineypos21), (0, 0, 255), 2)
                            caridscrossed1.append(
                                currentcarsindex1[i])  # adds car id to list of count cars to prevent double counting

                        # checks if old centroid is on or above line and curcent is on or below line
                        # to count cars and that car hasn't been counted yet
                        elif oldcent1[1] >= lineypos1 and curcent1[1] <= lineypos1 and carids1[
                            currentcarsindex1[i]] not in caridscrossed1:

                            carscrosseddown1 = carscrosseddown1 + 1
                            cv2.line(image1, (0, lineypos1), (width1, lineypos1), (0, 0, 125), 2)
                            caridscrossed1.append(
                                currentcarsindex1[i])

            cv2.putText(image1, "Cars Entered: " + str(3 + carscrossedup1), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, .37,
                        (0, 0, 255),
                        1)
            cv2.putText(image1, "Cars Left: " + str(carscrosseddown1), (0, 225), cv2.FONT_HERSHEY_SIMPLEX, .37,
                        (0, 0, 255),
                        1)
            totalcars = 3 + carscrossedup1 - carscrosseddown1
            cv2.putText(image1, "Total Cars: " + str(totalcars), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 0, 255),
                        1)

            # displays images and transformations
            cv2.namedWindow('bins', cv2.WINDOW_NORMAL)
            cv2.imshow("bins", bins1)

            cv2.namedWindow('countours', cv2.WINDOW_NORMAL)
            cv2.imshow("countours", image1)

            video1.write(image1)  # save the current image to video file from earlier

            # adds to framecount
            framenumber1 = framenumber1 + 1

            k = cv2.waitKey(int(1000 / fps1)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
            if k == 27:
                break

        else:  # if video is finished then break loop
            break

    cap1.release()
    cv2.destroyAllWindows()
    #df.to_csv('traffic.csv', sep=',')
    return True
