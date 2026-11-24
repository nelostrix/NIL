"""Local image semantics, budgets and complete classifier checkpoint workflow."""
import json
from pathlib import Path
import pytest
PIL=pytest.importorskip('PIL.Image')
torch=pytest.importorskip('torch')
import nil


def make_image(path, mode='RGB', color=(255,0,0), size=(7,5)):
    path.parent.mkdir(parents=True,exist_ok=True)
    PIL.new(mode,size,color).save(path)
    return path


def test_pixel_semantics_and_transform_roundtrip(tmp_path):
    transform=nil.vision.ImageTransform(size=(2,3))
    red=make_image(tmp_path/'red.png')
    x=transform(red)
    assert x.shape==(3,2,3) and x.dtype==torch.float32 and x.is_contiguous()
    torch.testing.assert_close(x[0],torch.ones(2,3))
    assert x[1:].count_nonzero()==0
    gray=transform(make_image(tmp_path/'gray.png','L',128))
    torch.testing.assert_close(gray,torch.full_like(gray,128/255))
    transparent=transform(make_image(tmp_path/'alpha.png','RGBA',(255,0,0,0)))
    torch.testing.assert_close(transparent,torch.ones_like(transparent))
    config=nil.vision.ImageTransform((2,3),mean=(.5,.5,.5),std=(.5,.5,.5))
    loaded=nil.vision.ImageTransform.from_dict(json.loads(json.dumps(config.to_dict())))
    torch.testing.assert_close(loaded(red),(x-.5)/.5)
    jpeg=make_image(tmp_path/'red.jpg')
    assert transform(jpeg)[0].mean()>.98


def test_exif_orientation_before_resize(tmp_path):
    path=tmp_path/'rotated.png'
    source=PIL.new('RGB',(2,1));source.putdata([(255,0,0),(0,0,255)])
    exif=source.getexif();exif[274]=6
    source.save(path,exif=exif)
    actual=nil.vision.ImageTransform((2,1))(path)
    torch.testing.assert_close(actual[:,0,0],torch.tensor([1.,0.,0.]))
    torch.testing.assert_close(actual[:,1,0],torch.tensor([0.,0.,1.]))


def test_corruption_frames_and_decode_budgets(tmp_path):
    path=make_image(tmp_path/'valid.png',size=(20,20))
    with pytest.raises(ValueError,match='max_pixels'):nil.vision.ImageTransform(max_pixels=10)(path)
    with pytest.raises(ValueError,match='max_bytes'):nil.vision.ImageTransform(max_bytes=10)(path)
    bad=tmp_path/'broken.png';bad.write_bytes(b'not an image')
    with pytest.raises(PIL.UnidentifiedImageError):nil.vision.ImageTransform()(bad)
    frames=tmp_path/'frames.tiff'
    PIL.new('RGB',(2,2)).save(frames,save_all=True,append_images=[PIL.new('RGB',(2,2),'white')])
    with pytest.raises(ValueError,match='multipage'):nil.vision.ImageTransform()(frames)


@pytest.mark.parametrize('options',[{'size':(0,1)}, {'size':(True,3)}, {'std':(1,1,1)},
    {'mean':(0,0,0),'std':(1,0,1)}, {'mean':(0,float('inf'),0),'std':(1,1,1)}, {'max_pixels':False}])
def test_invalid_preprocessing(options):
    with pytest.raises(ValueError):nil.vision.ImageTransform(**options)


def test_lazy_inventory_mapping_and_changes(tmp_path):
    make_image(tmp_path/'train'/'zebra'/'a.png')
    broken=tmp_path/'train'/'ant'/'b.png';broken.parent.mkdir();broken.write_bytes(b'broken')
    data=nil.vision.image_folders(tmp_path/'train')
    assert data.classes==('ant','zebra') and data.class_to_idx=={'ant':0,'zebra':1}
    assert len(data)==2 # indexing does not decode
    with pytest.raises(PIL.UnidentifiedImageError):data[0]
    make_image(broken,color=(0,255,0))
    assert data[0][1]==0
    make_image(tmp_path/'test'/'zebra'/'x.png')
    valid=nil.vision.image_folders(tmp_path/'test',classes=data.classes)
    assert valid[0][1]==1
    make_image(tmp_path/'test'/'unknown'/'x.png')
    with pytest.raises(ValueError,match='Unknown class'):nil.vision.image_folders(tmp_path/'test',classes=data.classes)
    with pytest.raises(ValueError,match='max_samples'):nil.vision.image_folders(tmp_path/'train',max_samples=1)
    broken.write_bytes(b'x'*200)
    bounded=nil.vision.image_folders(tmp_path/'train',transform=nil.vision.ImageTransform(max_bytes=100))
    with pytest.raises(ValueError,match='max_bytes'):bounded[0]


def test_train_evaluate_checkpoint_and_streaming_predictions(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(4)
    for label,color in [('red',(255,0,0)),('blue',(0,0,255))]:
        for i in range(4):make_image(tmp_path/'images'/label/f'{i}.png',color=color)
    data=nil.vision.image_folders(tmp_path/'images',transform=nil.vision.ImageTransform((8,8)))
    model=nil.model('nelo_vision_mini',num_classes=2,width=4)
    result=nil.train(model,data,task='classification',epochs=15,batch_size=8,learning_rate=.05,device='cpu',output_dir=tmp_path/'runs')
    assert nil.evaluate(result,data,task='classification',device_name='cpu')['accuracy']==1
    manifest=json.loads(json.dumps(data.manifest()))
    paths=[path for path,label in data.samples]
    predictions=list(nil.vision.classify(result,paths,manifest=manifest,batch_size=3))
    assert [r['label'] for r in predictions]==[data.classes[label] for path,label in data.samples]
    restored=nil.model('nelo_vision_mini',num_classes=2,width=4)
    nil.training.checkpoint.load(result.checkpoint,restored)
    assert list(nil.vision.classify(restored,paths,manifest=manifest,batch_size=3))==predictions
    assert all(sum(r['probabilities'])==pytest.approx(1) for r in predictions)
    assert result.model.training
    # A source iterator is consumed a batch at a time, not materialized eagerly.
    consumed=[]
    def source():
        for path in paths:consumed.append(path);yield path
    stream=nil.vision.classify(result,source(),manifest=manifest,batch_size=2)
    assert not consumed
    next(stream);assert len(consumed)==2
    stream.close()
    assert result.model.training
    with pytest.raises(ValueError,match='logits'):
        list(nil.vision.classify(torch.nn.Flatten(),paths[:1],manifest=manifest))
    with pytest.raises(ValueError,match='manifest'):
        list(nil.vision.classify(model,paths,manifest=dict(manifest,schema_version=True)))
