import extended_tools as TOOL

if __name__ == '__main__':
    TOOL.create_label_video_with_rotational_view(
        config_file_path='/home/cse509/DLCproj/E-cse509-E-2021-10-08-3d/config.yaml',
        video_folder_path='/home/cse509/DLCproj/E-videos',
        result_names=['projE_v01_DLC_3D', 'projE_v02_DLC_3D'],
        start_frame=100,
        frame_duration=200,
        start_view_horizontal=0,
        start_view_vertical=60,
        source_video_type='mp4',
        xlim=[None, None],
        ylim=[None, None],
        zlim=[None, None])
