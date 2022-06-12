import axios from "axios";
import DEPLOYED_UPLOAD_URL from "./constant"

let axiosConfig = {
    headers: {
        'Content-Type': 'application/json;charset=UTF-8',
        "Access-Control-Allow-Origin": "*",
    }
  };

export function getImageList() {
    return axios.get(DEPLOYED_UPLOAD_URL,axiosConfig)
            .then(image => {
                console.log('Image List: ', image.data);
                return image.data
            })
}

export function updateUploadById (id, data){
    const response =  axios.put(DEPLOYED_UPLOAD_URL + '/' + id, data, axiosConfig)
    return response
}
