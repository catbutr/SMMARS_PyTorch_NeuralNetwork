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

def file_callback(sender, app_data):
    print('OK was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)

def file_cancel_callback(sender, app_data):
    print('Cancel was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)

with dpg.file_dialog(directory_selector=False, show=False, callback=file_callback, cancel_callback=file_cancel_callback, id="file_dialog_id", width=700 ,height=400):
    dpg.add_file_extension(".csv", color=(255, 0, 255, 255), custom_text="[CSV]")
    dpg.add_file_extension(".xlsl", color=(0, 255, 0, 255), custom_text="[Excel]")

with dpg.window(label="Tutorial", width=400, height=400):

    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback):
        with dpg.node(label="CSV1"):
            with dpg.node_attribute(label="read_from_file", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_button(label="Open file",callback=lambda:dpg.show_item("file_dialog_id"))

        with dpg.node(label="CSV2"):
            with dpg.node_attribute(label="read_from_file", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_button(label="Open file",callback=lambda:dpg.show_item("file_dialog_id"))

        with dpg.node(label="Splitter"):
            with dpg.node_attribute(label="X"):
                dpg.add_text("Features")
            with dpg.node_attribute(label="Y"):
                dpg.add_text("Labels")
            with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("Test1")
            with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("Test2")
            with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("Train1")
            with dpg.node_attribute(label="output", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("Train2")
                
        with dpg.node(label="Training"):
            with dpg.node_attribute(label="input", attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text("Test1")
            with dpg.node_attribute(label="input", attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text("Test2")
            with dpg.node_attribute(label="input", attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text("Train1")
            with dpg.node_attribute(label="input", attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text("Train2")
            with dpg.node_attribute(label="output",attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("Error")
            with dpg.node_attribute(label="output",attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("Weights")

dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()