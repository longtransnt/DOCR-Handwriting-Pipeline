import './App.css';
import 'react-dropdown/style.css';
import 'react-toastify/dist/ReactToastify.css';

import Form from 'react-bootstrap/Form'
import React, { useState, useCallback, useEffect } from "react";
import ListGroup from 'react-bootstrap/ListGroup'
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'
import Stack from 'react-bootstrap/Stack'
import ImageUpload from './components/ImageUpload';
import { ToastContainer, toast } from 'react-toastify';
import { Scrollbars } from 'react-custom-scrollbars'
import { Dropdown } from 'react-bootstrap'
import { IoChevronDown } from 'react-icons/io5'
import getImageList from './service';

let verified = {};
let confidenceValue = {}

const notiSaving = () => toast.warn('Please input annotation before saving!', {
  position: "top-right",
  autoClose: 2000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  progress: undefined,
});

const notiDownload = () => toast.warn('Required at least 1 annotation to download!', {
  position: "top-right",
  autoClose: 2000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  progress: undefined,
});
// Main application
function App() {
  const [currId, setCurrId] = useState(0);
  const [currImagePath, setCurrImagePath] = useState('')
  const [annotation, setAnnotation] = useState('');
  const [annotationList, setAnnotationList] = useState([]);
  const [updateState, setUpdateState] = useState(0);
  const [checked, setChecked] = useState(false);
  const [image, setImage] = useState([]);
  const [confidenceState, setConfidenceState] = useState('');

  
  const fetchUploads = useCallback(() => {
    getImageList().then(data => {setImage(data)})
      .catch(console.error)
  }, []);

  useEffect(() => {
    fetchUploads();
  }, [fetchUploads])

  const handleAdd = () => {
    console.log(updateState);
    if (annotation !==  '') {
      if (updateState === 1 ) {
        annotationList[currId] = image[currId] + "\t" + annotation + "\n";
        verified[currId] = checked;
        confidenceValue[currId] = confidenceState;
        setUpdateState(0);
      }
      else {
        // Put function to save new annotation here
        const newList = annotationList.concat(image[currId] + "\t" + annotation + "\n")
        setAnnotationList(newList);
      }
    // Move to next image
    setAnnotation('')
    setCurrId(currId + 1);
    } else {
      setUpdateState(0);
      notiSaving(); // not allow to save if no annotation
    }
  }

  const handleListClick = (id) => {
     // Move to this image
     setCurrId(id);
     setCurrImagePath(image[id].imageUrl)
     setAnnotation("");
     setUpdateState(1);
     setChecked(false);
     setConfidenceState('');
  }

  function WriteToFile() {
    const element = document.createElement("a");
    console.log(annotationList);
    if (annotationList.length === 0) {
      notiDownload()
    } else {
      var data = annotationList.filter(function( element ) {
        return (element !== undefined && verified[annotationList.indexOf(element)]); // only verified annotation can be downloaded
      });
      const file = new Blob(data, {
        type: "text/plain"
      });
      element.href = URL.createObjectURL(file);
      element.download = "annotation.txt";
      document.body.appendChild(element);
      element.click();
    }
  }

  const handleChecked = () => {
    setChecked(!checked)
  };

  const handleConfidenceSelect = (value) => {
    setConfidenceState(value)
  };

  return (
    <div className="App">
      <div className="App-header">
        <button className='upload-btn'>Upload</button>
        <Container>
          <Row xs={1} md={2}>
            <Col>
              {/* Displaying Image */}
              <p style={{textAlign: 'center'}}>Current Image</p>
              <Stack gap={4} className="col-md-11 mx-auto">
                <div style={{display: 'flex', justifyContent: 'center'}}>
                  <img 
                    className='img-display'
                    id={currId} 
                    src={currImagePath}
                  />
                </div>
                <p style={{fontSize: '22px'}}>Annotation Preview: 
                  <span style={{fontSize: '18px', paddingLeft: '5px'}}>
                    {/* Preview annotation */}
                    {annotationList[currId] !== undefined ? annotationList[currId].split(image[currId] + "\t") : annotation !== '' ? annotation : "None"}
                  </span>
                </p>
                <Form onSubmit={handleAdd}>
                  <Form.Group className="mb-3" controlId="formBasicEmail">
                    <Form.Control 
                    as="textarea" 
                    rows={4} 
                    placeholder="Enter your annotation here." 
                    style={{border: 'none', height: '20vh'}}
                    value={annotation} 
                    onChange={(e) => {setAnnotation(e.target.value)}}
                    />
                    <div style={{display: 'flex', justifyContent: 'space-between'}}>
                        <div style={{display: 'flex', alignItems: 'center'}}>
                          <div>
                            <label style={{fontSize: '1rem', color: '#005477', float: 'left', marginRight: '5px'}}>
                              % Confidence
                            </label>
                          </div>
                          <div>
                            <Dropdown onSelect={handleConfidenceSelect}>
                              <Dropdown.Toggle id="dropdown-split-basic">
                                {confidenceState === '' ? '100%' : confidenceState}
                                <IoChevronDown style={{width: '1rem', height: '1rem', marginLeft: '5px'}}/>
                              </Dropdown.Toggle>
                              <Dropdown.Menu>
                                <Dropdown.Item className='dropdown-item' eventKey="100%">100%</Dropdown.Item>
                                <Dropdown.Item className='dropdown-item' eventKey="75%">75%</Dropdown.Item>
                                <Dropdown.Item className='dropdown-item' eventKey="50%">50%</Dropdown.Item>
                                <Dropdown.Item className='dropdown-item' eventKey="25%">25%</Dropdown.Item>
                              </Dropdown.Menu>
                            </Dropdown>
                          </div>
                        </div>
                        <div style={{display: 'flex', alignItems: 'center', fontSize: '1rem', color: '#005477'}}>
                          <div>
                            <input 
                              id='checkbox3'
                              style={{margin: '5px 5px 0 0', color: '#005477', fontWeight: '500', cursor: 'pointer'}} 
                              type="checkbox"
                              checked={checked} onChange={handleChecked}
                            />
                          </div>
                          <div>
                            <label htmlFor="verified" onClick={handleChecked} style={{cursor: 'pointer'}}>Verified by OUCRU</label>
                          </div>
                        </div>
                      </div>
                  </Form.Group>
                </Form>
              </Stack>
            </Col>
            <Col>
              {/* Image List */}
              <p style={{textAlign: 'center'}}>Image List</p>
              <Scrollbars>
                <div id="image-list">
                  {image.map((im, id) => (
                      <ListGroup.Item 
                        id={"image_" + id} 
                        key={id} 
                        value={id}
                        variant={
                          annotationList[id] === undefined ? "danger" : 
                          verified[id] === false ? "warning" : "success"
                        } 
                        style={{cursor: 'pointer'}}
                        onClick={(e) => {handleListClick(id)}}
                      >
                        {im.file_name && im.file_name.length ? (
                          <img src= {im.thumbnailUrl} />
                        ) : null}
                        {im.file_name}
                      </ListGroup.Item>
                  ))}
                </div>
              </Scrollbars>
            </Col> 
          </Row>
        </Container>
        <Row style={{marginTop: '7rem'}}>
          <Col style={{position : 'fixed', bottom: 0, marginBottom: '1rem'}}>
            <div style={{float: 'left'}}>
              <ImageUpload />
            </div>
            <div style={{float: 'right', marginRight: '46px'}}>
              <button className='save-btn' onClick={handleAdd}>Save the annotation</button>{' '}
              <div style={{float: 'right'}}>
                <Dropdown>
                  <Dropdown.Toggle id="dropdown-basic-button">
                    DOWNLOAD
                    <IoChevronDown style={{width: '1.5rem', height: '1.5rem', marginLeft: '5px'}}/>
                  </Dropdown.Toggle>
                  <Dropdown.Menu>
                    <Dropdown.Item className='dropdown-item' eventKey="25%">25% Confidence</Dropdown.Item>
                    <Dropdown.Item className='dropdown-item' eventKey="50%">50% Confidence</Dropdown.Item>
                    <Dropdown.Item className='dropdown-item' eventKey="75%">75% Confidence</Dropdown.Item>
                    <Dropdown.Item className='dropdown-item' eventKey="100%">100% Confidence</Dropdown.Item>
                    <Dropdown.Divider />
                    <Dropdown.Item className='dropdown-item' eventKey="Download all" onClick={WriteToFile}>Download All</Dropdown.Item>
                  </Dropdown.Menu>
                </Dropdown>
              </div>
            </div>
          </Col>
        </Row>
      </div>
      <ToastContainer style={{width: '20vw'}} />
    </div>
  );
}

export default App;
