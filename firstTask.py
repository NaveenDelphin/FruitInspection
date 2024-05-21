import cv2
import numpy as np

def run1():
   
    picNo = input("Enter the picture number(1, 2 or 3): ")
    # Load NIR and color images
    nir_image = cv2.imread('Task1pics/C0_00000' + picNo +'.png', cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread('Task1pics/C1_00000' + picNo +'.png')
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    #Application of a Gaussian Blur
    blur_temp = cv2.GaussianBlur(nir_image,(3,3),0)
    cv2.imshow('bluredImage', blur_temp)

    #Otsu Thresholding to segment the image
    th, otsu_temp = cv2.threshold(blur_temp, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    cv2.imshow('ThresholdedImage', otsu_temp)

    #filling of the holes inside the fruit blob using a flood-fill approach
    h, w = otsu_temp.shape[:2]
    m1 = np.zeros((h+2, w+2), np.uint8)
    ff1 = otsu_temp.copy()
    cv2.floodFill(ff1, m1, (0,0), 255)
    #we then invert the result obtained by the floodfill operation in order to highlight the holes
    holes_temp = cv2.bitwise_not(ff1)
    cv2.imshow('FilledHoles', holes_temp)

    #Mask of the apples
    mask_temp = holes_temp | otsu_temp
    mask_c_temp = cv2.cvtColor(mask_temp, cv2.COLOR_GRAY2RGB) 
    cv2.imshow('Mask', mask_c_temp)

    #Application of the masks to infrared images
    app_nir_temp = nir_image * (mask_temp/255)
    cv2.imshow('nirMasked', app_nir_temp)

    #Application of the masks to colored images
    ones = np.ones(rgb_image.shape, dtype=int)
    bool_mask_temp = ones & mask_c_temp
    app_rgb = rgb_image * bool_mask_temp.astype(np.uint8)
    cv2.imshow('MaskedImage', app_rgb)

    #Application of Canny Edge Detector after a Bilateral Filter
    app_nir_temp = app_nir_temp.astype(np.uint8)
    img_bilateral = cv2.bilateralFilter(app_nir_temp,11,35,35)
    edges_temp = cv2.Canny(img_bilateral, 50, 130)
    cv2.imshow('edge', edges_temp)

    #Perform a closing operation to have better defects' edges
    clo_ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    edges_temp = edges_temp.astype(np.uint8)
    closing_temp=cv2.morphologyEx(edges_temp, cv2.MORPH_CLOSE, clo_ker)

    #Fill of the undefected parts of the apples
    h_ff, w_ff= closing_temp.shape
    closing_temp = closing_temp.astype(np.uint8)
    row_ff=np.zeros((2, w_ff))
    col_ff=np.zeros((h_ff+2, 2))
    n_mask=~mask_temp
    mask_ff_temp=np.vstack((n_mask, row_ff))
    mask_ff_temp=np.hstack((mask_ff_temp, col_ff))
    mask_ff_temp = mask_ff_temp.astype(np.uint8)
    cv2.floodFill(closing_temp, mask_ff_temp, (175,150), 255)
    holes_ff_temp= closing_temp
    cv2.imshow("filled", holes_ff_temp)

    #Subtract the previous image to the fully filled mask of the apple to higlight the defects
    open_ker=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    holes_ff_temp = holes_ff_temp.astype(np.uint8)    
    holes_ff_temp=~holes_ff_temp
    sub_temp=mask_temp & holes_ff_temp
    open_temp=cv2.morphologyEx(sub_temp, cv2.MORPH_OPEN, open_ker)
    cv2.imshow('defectsAlone', open_temp)

    # Find contours of the defects in open_temp
    contours, _ = cv2.findContours(open_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw rounded contours
    rounded_contours_img = np.zeros_like(color_image)

    # Draw rounded contours around the defects
    for contour in contours:
        # Calculate center and radius of the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        # Draw the circle around the contour
        cv2.circle(rounded_contours_img, center, radius, (0, 255, 0), 2)

    # Superimpose the rounded contours onto the original color image
    result_image = cv2.addWeighted(color_image, 1, rounded_contours_img, 0.5, 0)

    # Display the result
    cv2.imshow('Defects on Color Image', result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
