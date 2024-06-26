##################################################################
### Validate prediction with field estimations by fieldworkers ###
##################################################################

validate(field_data, # (str) path to field estimations of coverage by fieldworkers
         predictions, # (str) path to predicted coverage
         image_ID_column, # (str) column which exists in both datasets (Image ID)
         classes_to_include, # (lst) column names of classes to include in the validation (must be the same in field and prediction dataset)
         names_of_classes, # (lst) names to be used in the legend of the scatterplot
         colors) # (lst) colours to be used in the scatterplot
