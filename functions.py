######################################################################################################################
### function to predict with frame detection and species segmentation model to get percentage coverage per species ###
######################################################################################################################

def predict(path_to_data_images, path_to_model_frame=trained_model_path_frame, name_frame_class="frame",  # parameter for cropping to frame
            path_to_model_species=trained_model_path_species, conf_treshold=0.1,                          # parameter for species segmentation
            number_of_classes=2):                                                                         # parameter for calculating percentage cover

    """
    Predict with YOLO models with the specified parameters:

    Parameters for predicting with frame detection model:
    - path_to_data_images (str): Path to the folder with orignal images taken in the field.
    - path to model  frame (str): Path to the YOLO model which crops orignal images to the inner frame.
    - name_frame_class (str): Name of the class representing the frame in the frame detection model. Default is "frame".

    Parameters for predicting with species segmentation model:
    - path_to_model_species (str):  Path to the YOLO model which segments species.
    - conf_thresold (int): Confidence threshold for the prediction. Default is 0.1.

    Parameters for calculating percentage cover of the species:
    - number_of_classes (int): Total number of classes in species segmentation model. Default is 2.

    """

    """ Crop original images """

    model_frame = YOLO(path_to_model_frame)
    model_frame(path_to_data_images,
                save=True,
                project=path_to_data_images,
                name="temp_cropped_images",
                max_det=1,
                save_crop=True)

    cropped_images_path = os.path.join(path_to_data_images, "temp_cropped_images", "crops", name_frame_class)

    """ Species segmentation on cropped images """

    model_species = YOLO(path_to_model_species)
    model_species(cropped_images_path,
                  conf=conf_treshold,
                  #agnostic_nms = True,
                  save=True,
                  save_txt=True,
                  project=path_to_data_images,
                  name="temp_species_segmentation")

    predicted_images_path = os.path.join(path_to_data_images, "temp_species_segmentation")
    predicted_images_path_labels = os.path.join(predicted_images_path, "labels")

    """ Create grayscale images """

    class_intensity_mapping = {}
    used_grayscale_values = set()

    for class_id in range(number_of_classes):
        grayscale_value = random.randint(0, 255)
        while grayscale_value in used_grayscale_values:
            grayscale_value = random.randint(0, 255)
        class_intensity_mapping[class_id] = grayscale_value
        used_grayscale_values.add(grayscale_value)

    intensity_to_class = {v: k for k, v in class_intensity_mapping.items()}

    results = []

    for image_filename in os.listdir(predicted_images_path):
        if image_filename.endswith(".jpg"):
            image_path = os.path.join(predicted_images_path, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(predicted_images_path_labels, label_filename)

            image = cv2.imread(image_path)
            height, width, _ = image.shape
            accumulated_mask = np.zeros((height, width), dtype=np.uint8)

            with open(label_path, "r") as file:
                for line in file:
                    class_number = int(line[0])
                    object_info = line.split()[1:]
                    object_info = list(map(float, object_info))
                    coordinates_pairs = np.array(object_info).reshape(-1, 2)
                    coordinates_pairs[:, 0] *= width
                    coordinates_pairs[:, 1] *= height
                    intensity = class_intensity_mapping.get(class_number, 0)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [coordinates_pairs.astype(int)], intensity)
                    accumulated_mask = cv2.bitwise_or(accumulated_mask, mask)

            total_pixels = accumulated_mask.size
            unique_classes = np.unique(accumulated_mask)
            class_pixel_shares = {'Image': image_filename}

            for target_class in unique_classes:
                if target_class == 0:
                    continue
                class_pixels = np.sum(accumulated_mask == target_class)
                pixel_share = (class_pixels / total_pixels) * 100
                class_id = intensity_to_class.get(target_class, "mixed class")
                class_pixel_shares[f'Class_{class_id}'] = pixel_share

            results.append(class_pixel_shares)

    results_df = pd.DataFrame(results)
    results_folder = os.path.join(path_to_data_images, "Results")
    os.makedirs(results_folder, exist_ok=True)
    results_file_path = os.path.join(results_folder, 'results.csv')
    results_df.to_csv(results_file_path, index=False)

    print(f"Results saved to: {results_file_path}")

    # Clean up temporary folders if wanted
    shutil.rmtree(os.path.join(path_to_data_images, "temp_cropped_images"))
    shutil.rmtree(os.path.join(path_to_data_images, "temp_species_segmentation"))

############################################
### Evaluate predictions with field data ###
############################################

def validate(field_data, # (str) path to field estimations of coverage by fieldworkers
             predictions, # (str) path to predicted coverage
             image_ID_column, # (str) column which exists in both datasets (Image ID)
             classes_to_include, # (lst) column names of classes to include in the validation (must be the same in field and prediction dataset)
             names_of_classes, # (lst) names to be used in the legend of the scatterplot
             colors): # (lst) colours to be used in the scatterplot

    # Initialize lists for RMSE and R-squared values
    rmse_list = []
    r_squared_list = []
    md_list = []

    # Create a scatterplot for each class
    for i, class_name in enumerate(classes_to_include):
        # Merge dataframes based on the 'Image' column
        merged_df = pd.merge(field_data[[image_ID_column, class_name]], 
                             predictions[[image_ID_column, class_name]], 
                             on=image_ID_column, 
                             suffixes=('_field', '_pred'))

        # Convert columns to numeric (if not already)
        merged_df[f'{class_name}_field'] = pd.to_numeric(merged_df[f'{class_name}_field'], errors='coerce')
        merged_df[f'{class_name}_pred'] = pd.to_numeric(merged_df[f'{class_name}_pred'], errors='coerce')

        # Scatterplot for the current class
        plt.scatter(merged_df[f'{class_name}_field'], merged_df[f'{class_name}_pred'], label=names_of_classes[i], color=colors[i])

        # Calculate RMSE for the current class
        rmse = np.sqrt(np.mean((merged_df[f'{class_name}_field'] - merged_df[f'{class_name}_pred'])**2))
        rmse_list.append(rmse)

        # Calculate R-squared for the current class
        ssr = np.sum((merged_df[f'{class_name}_pred'] - merged_df[f'{class_name}_field'])**2) 
        sst = np.sum((merged_df[f'{class_name}_field'] - np.mean(merged_df[f'{class_name}_field']))**2)

        r_squared = 1 - (ssr / sst)
        r_squared_list.append(r_squared)

        # Calculate MD
        md = np.mean(merged_df[f'{class_name}_field'] - merged_df[f'{class_name}_pred'])
        md_list.append(md)

    # Plot the identity line
    min_val = 0
    max_val = 100
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black')

    # Set labels and title
    plt.xlabel('Coverage estimated by fieldworker [%]')
    plt.ylabel('Predicted coverage [%]')
    plt.title('Percentages of coverage fieldwork vs. predicted')

    # Create a custom legend with desired labels
    plt.legend(loc='upper left')

    # Get the current axis
    ax = plt.gca()

    # Show the plot
    plt.show()

    # Display RMSE and R-squared values
    for i, class_name in enumerate(classes_to_include):
        print(f'RMSE for {class_name}: {rmse_list[i]}')
        print(f'R-squared for {class_name}: {r_squared_list[i]}')
        print(f'MD for {class_name}: {md_list[i]}')
