import logo from './logo.svg';
import './App.css';
import Form from 'react-bootstrap/Form'
import React, { useState } from "react";
import Button from 'react-bootstrap/Button'
import { v4 as uuidv4 } from 'uuid';

let imageList = {};

// Import all images in data folder
function importAll(r) {
  imageList = r.keys();
  return r.keys().map(r);
}

const images = importAll(require.context('./../data', false, /\.(png|jpe?g|svg)$/));

// Main application
function App() {
  const [currImage, setCurrImage] = useState(0);
  const [annotation, setAnnotation] = useState('');
  const [annotationList, setAnnotationList] = React.useState([]);

  const handleAdd = () => {
    // Put function to save annotation here
    const newList = annotationList.concat(imageList[currImage] + " " + annotation + "\n")
    setAnnotationList(newList);

    // Move to next image
    setAnnotation('')
    setCurrImage(currImage + 1);
  }

  function WriteToFile() {
    const element = document.createElement("a");
    const file = new Blob(annotationList, {
      type: "text/plain"
    });
    element.href = URL.createObjectURL(file);
    element.download = "annotation.txt";
    document.body.appendChild(element);
    element.click();
  }

  return (
    <div className="App">
      <header className="App-header">
      <p>Image Annotation</p>
    
      {/* Displaying Image */}
      <img src= {images[currImage]} />

      {/* Form for text input */}
      <Form onSubmit={handleAdd}>
      <Form.Group className="mb-3" controlId="formBasicEmail">
        <br/>
        <Form.Label>Enter annotation</Form.Label>
        <br/>
        <Form.Control as="textarea" rows={4} value={annotation} onChange={(e) => {setAnnotation(e.target.value)}}/>
      </Form.Group>
      </Form>
      <p>{annotation}</p>
      <br/>
      {/* Button to save annotation */}
      <Button variant="primary" onClick={handleAdd}>Save this annotation</Button>{' '}
      <Button variant="primary" onClick={WriteToFile}>Download all annotation</Button>{' '}

      </header>
    </div>
  );
}

export default App;

