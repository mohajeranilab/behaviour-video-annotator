"""Label Studio SDK wrapper for project management and task operations."""

import time
from label_studio_sdk import LabelStudio


LABELING_CONFIG = """
<View style="display: flex; flex-direction: row; gap: 24px; padding: 10px;">
  
  <!-- LEFT: The Video Player (70% width) -->
  <View style="flex: 70%;">
    <Video name="video" value="$video" />
  </View>
  
  <!-- RIGHT: Categories Panel (30% width) -->
  <View style="flex: 30%;">
    <Header value="Select All Categories That Apply" />
    <!-- Changing 'choice' parameter to 'multiple' opens checkbox behaviors -->
    <Choices name="category" toName="video" choice="multiple">
      <Choice value="Resting" hotkey="1, t" />
      <Choice value="Walking" hotkey="2, w" />
      <Choice value="Rearing" hotkey="3, r" />
      <Choice value="Nesting" hotkey="4, n" />
      <Choice value="Sniffing" hotkey="5, s" />
      <Choice value="Glooming" hotkey="6, g" />
      <Choice value="Drinking" hotkey="7, d" />
      <Choice value="Eating" hotkey="8, e" />
      <Choice value="Other" hotkey="9, o" />
    </Choices>
  </View>

  <!-- Safely hides the native outliner sidebar from view -->
  <Style>
    .lsf-main-view__sidebar { 
      display: none !important; 
    }
  </Style>
  
</View>
""".strip()

PROJECT_NAME = "Mouse Behavior Annotation"


def get_client(base_url: str, api_key: str) -> LabelStudio:
    """Create and return a Label Studio SDK client."""
    return LabelStudio(base_url=base_url, api_key=api_key)


def ensure_project(client: LabelStudio, project_name: str = PROJECT_NAME) -> int:
    """Find or create the annotation project. Returns project ID."""
    projects = client.projects.list()
    for p in projects:
        if p.title == project_name:
            print(f"  Found existing project: {project_name} (id={p.id})")
            
            # Update label config if it has changed to match new custom UI
            if p.label_config.strip() != LABELING_CONFIG:
                print("  Updating project labeling config with new layout...")
                client.projects.update(id=p.id, label_config=LABELING_CONFIG)
                
            return p.id

    # Create new project
    project = client.projects.create(
        title=project_name,
        label_config=LABELING_CONFIG,
    )
    print(f"  Created project: {project_name} (id={project.id})")
    
    # Automatically configure Local Storage for the project so the clips are accessible
    try:
        client.import_storage.local.create(
            project=project.id,
            path='/label-studio/clips',
            use_blob_urls=True,
            title='Clips Storage'
        )
        print(f"  Successfully configured Local Storage (/label-studio/clips) for project {project.id}")
    except Exception as e:
        print(f"  Warning: Failed to configure Local Storage automatically: {e}")
        
    return project.id


def push_clips(
    client: LabelStudio,
    project_id: int,
    clip_metadata: list[dict],
    wave_id: int,
) -> list[int]:
    """Create tasks in Label Studio from clip metadata.

    Each clip_metadata dict must have 'filename' and 'predicted_label'.
    The video URL uses Label Studio's local file serving format.

    Returns list of created task IDs.
    """
    tasks = []
    for clip in clip_metadata:
        tasks.append({
            'data': {
                'video': f"/data/local-files/?d=clips/{clip['filename']}",
                'predicted_label': clip['predicted_label'],
                'video_prefix': clip.get('video_prefix', ''),
                'start_sec': clip.get('start_sec', 0),
                'end_sec': clip.get('end_sec', 0),
                'wave_id': wave_id,
            }
        })

    if not tasks:
        return []

    client.projects.import_tasks(id=project_id, request=tasks)
    
    # Fetch all tasks in the project and filter by wave_id to get the newly created IDs
    all_tasks = list(client.tasks.list(project=project_id))
    task_ids = []
    for t in all_tasks:
        t_data = t.get('data', {}) if isinstance(t, dict) else getattr(t, 'data', {})
        if not isinstance(t_data, dict):
            t_data = getattr(t_data, '__dict__', {}) if t_data else {}
        if t_data.get('wave_id') == wave_id:
            tid = t.get('id') if isinstance(t, dict) else getattr(t, 'id', None)
            if tid is not None:
                task_ids.append(tid)
    
    print(f"  Pushed {len(tasks)} tasks to project {project_id} (wave {wave_id})")
    return task_ids


def get_completed_annotations(
    client: LabelStudio,
    project_id: int,
    task_ids: list[int],
) -> list[dict]:
    """Retrieve annotations for the specified task IDs.

    Returns list of dicts: {task_id, filename, predicted_label, annotated_label, video_prefix, ...}
    """
    results = []
    for tid in task_ids:
        try:
            task = client.tasks.get(id=tid)
        except Exception:
            continue

        annotations = task.get('annotations', None) if isinstance(task, dict) else getattr(task, 'annotations', None)
        if not annotations:
            continue

        annotation = annotations[0]
        selected_choices = None
        
        # Support both dict and object structures from Label Studio SDK
        res_list = annotation.get('result', []) if isinstance(annotation, dict) else getattr(annotation, 'result', [])
        
        for r in res_list:
            from_name = r.get('from_name') if isinstance(r, dict) else getattr(r, 'from_name', None)
            if from_name == 'category':
                val = r.get('value', {}) if isinstance(r, dict) else getattr(r, 'value', {})
                selected_choices = val.get('choices', []) if isinstance(val, dict) else getattr(val, 'choices', [])
                break

        if selected_choices is None:
            continue

        # If 'Resting' is one of the chosen labels, treat it as Sleeping.
        # Otherwise, treat it as Awake.
        mapped_label = 'Sleeping' if 'Resting' in selected_choices else 'Awake'

        task_data = task.get('data', {}) if isinstance(task, dict) else getattr(task, 'data', {})
        if not isinstance(task_data, dict):
            task_data = getattr(task_data, '__dict__', {}) if task_data else {}

        video_url = task_data.get('video', '')
        # Extract the pure filename (e.g. 20260410_045000_dca632e87112_awake_000090_000180.mp4)
        # from the Label Studio serving URL (/data/local-files/?d=clips/...)
        clean_filename = video_url.split('clips/')[-1] if 'clips/' in video_url else video_url.split('/')[-1]

        prefix = task_data.get('video_prefix', '')
        if not prefix and clean_filename:
            # Fallback: if video_prefix was not explicitly saved in task metadata,
            # reconstruct from filename format: <date>_<time>_<camera>_<label>_<start>_<end>.mp4
            parts = clean_filename.split('_')
            if len(parts) >= 3:
                prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"

        results.append({
            'task_id': tid,
            'video_url': video_url,
            'filename': clean_filename,
            'predicted_label': task_data.get('predicted_label', ''),
            'video_prefix': prefix,
            'start_sec': task_data.get('start_sec', 0),
            'end_sec': task_data.get('end_sec', 0),
            'annotated_label': mapped_label,
            'raw_annotated_labels': selected_choices,
        })

    return results


def check_wave_complete(
    client: LabelStudio,
    project_id: int,
    task_ids: list[int],
) -> bool:
    """Check if all tasks in a wave have been annotated."""
    if not task_ids:
        return True

    for tid in task_ids:
        try:
            task = client.tasks.get(id=tid)
            annotations = task.get('annotations', None) if isinstance(task, dict) else getattr(task, 'annotations', None)
            if not annotations:
                return False
        except Exception as e:
            print(f"  [check_wave_complete] Task {tid} lookup failed (404/error): {e}")
            return False

    return True
