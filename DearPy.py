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

with dpg.window(label="Tutorial", width=400, height=400):

    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback):
        with dpg.node(label="FeedForward NN"):
            with dpg.node_attribute(label="Input"):
                dpg.add_input_int(label="Input size", width=150)
                dpg.add_input_int(label="Input size", width=150)
                dpg.add_input_int(label="Hidden size", width=150)
                dpg.add_input_int(label="Output size", width=150)
                dpg.add_input_int(label="Number of layers", width=150)
                dpg.add_combo(label="Activation Function", items=['ReLU','LReLU','Sigmoid','Tahn','SoftMax'])


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()