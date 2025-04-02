import dearpygui.dearpygui as dpg

dpg.create_context()

# callback runs when user attempts to connect attributes
def link_callback(sender, app_data):
    # app_data -> (link_id1, link_id2)
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)

# callback runs when user attempts to disconnect attributes
def delink_callback(sender, app_data):
    # app_data -> link_id
    dpg.delete_item(app_data)

def callback(sender, app_data, user_data):
    print("Sender: ", sender)
    print("App Data: ", app_data)

def FNN():
    with dpg.node(label="FeedForward NN"):
        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_input_int(label="Input size", width=150, tag="InputSize")
            dpg.add_input_int(label="Hidden size", width=150,tag="HiddenSize")
            dpg.add_input_int(label="Output size", width=150,tag="OutputSize")
            dpg.add_input_int(label="Number of layers", width=150,tag="NumberOfLayers")
            dpg.add_combo(label="Activation Function", items=['ReLU','LReLU','Sigmoid','Tahn','SoftMax'], width=150, tag="ActivationFunction")

def File_Reader():
    with dpg.node(label="File Reader"):
        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_button(label="File Selector", callback=lambda: dpg.show_item("file_dialog_id"), tag="FileSelectorFile")

def Splitter():
    with dpg.node(label="Splitter"):
        with dpg.node_attribute(label="Input"):
            dpg.add_input_text(width=150, hint="Labels", tag = "InputLabels")
            dpg.add_input_text(width=150, hint="Features", tag="InputFeatures")
            dpg.add_input_double(label="Test Size",width=150, tag="TestSize")
            dpg.add_input_int(label="Random State",width=150, tag="RandomState")
        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Train_Features", tag="TrainFeatures")
        with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Train_Labels", tag="TrainLabels")
        with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Test_Features", tag="TestFeatures")
        with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Test_Labels", tag="TestLabels")

def DataLoader(loaderType):
    with dpg.node(label=loaderType):
        with dpg.node_attribute(label="Input"):
            dpg.add_text("features_input", tag="FeaturesInput")
        with dpg.node_attribute(label="Input"):
            dpg.add_text("labels_input", tag="LabelsInput")
        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("DataLoader", tag="DataLoader")

def Train():
    with dpg.node(label="Trainer"):
        with dpg.node_attribute(label="Input"):
            dpg.add_text("Model", tag="ModelTrain")
        with dpg.node_attribute(label="Input"):
            dpg.add_text("Train DataLoader", tag="TrainDataloader")
        with dpg.node_attribute(label="Output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_input_int(label="Number of epochs", width=150, tag="NumberOfEpochs")

def Predict():
    with dpg.node(label="Predict"):
        with dpg.node_attribute(label="Input"):
            dpg.add_text("Model", tag="ModelPredict")
        with dpg.node_attribute(label="Input"):
            dpg.add_text("Test DataLoader", tag="TestDataLoader")
        with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
            dpg.add_text("Prediction results", tag="PredictionResults")

# creating data
sindatax = []
sindatay = []
for i in range(0, 500):
    sindatax.append(i / 1000)
    sindatay.append(0.5 + 0.5)


with dpg.file_dialog(directory_selector=False, show=False, callback=callback, id="file_dialog_id", width=700 ,height=400):
    dpg.add_file_extension(".csv", color=(255, 0, 255, 255), custom_text="[CSV]")
    dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255), custom_text="[Excel]")

with dpg.window(label="Tutorial", width=400, height=400):
    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback):
        #Блок чтения файла
        File_Reader()
        #Блок нейронной сети
        FNN()
        Splitter()
        DataLoader("TrainLoader")
        DataLoader("TestLoader")
        Train()
        Predict()


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()