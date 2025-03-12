from .gradio_tasks import dense_prediction_text, conditional_generation_text, process_dense_prediction_tasks, process_conditional_generation_tasks
from .gradio_tasks_restoration import image_restoration_text, process_image_restoration_tasks
from .gradio_tasks_style import style_transfer_text, style_condition_fusion_text, process_style_transfer_tasks, process_style_condition_fusion_tasks
from .gradio_tasks_tryon import tryon_text, process_tryon_tasks
from .gradio_tasks_editing import editing_text, process_editing_tasks
from .gradio_tasks_photodoodle import photodoodle_text, process_photodoodle_tasks
from .gradio_tasks_editing_subject import editing_with_subject_text, process_editing_with_subject_tasks
from .gradio_tasks_relighting import relighting_text, process_relighting_tasks
from .gradio_tasks_unseen import unseen_tasks_text, process_unseen_tasks
from .gradio_tasks_subject import subject_driven_text, condition_subject_fusion_text, condition_subject_style_fusion_text, style_transfer_with_subject_text, \
    image_restoration_with_subject_text, \
    process_subject_driven_tasks, process_image_restoration_with_subject_tasks, process_style_transfer_with_subject_tasks, process_condition_subject_style_fusion_tasks, \
    process_condition_subject_fusion_tasks