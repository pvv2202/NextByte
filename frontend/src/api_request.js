
// TODO: change upon publishing
const API_URL = 'http://127.0.0.1:5000/api'

export const api_request = async (endpoint, method, headers, body) => {
    try{
      const response = await fetch(`${API_URL}/${endpoint}`, {
        method: method,
        credentials: 'include',
        headers: {...headers},
        body: JSON.stringify(body)
      })
      if (!response.ok) {
        let errorMsg = 'Error fetching data'; 
        const errData = await response.json();
        errorMsg = errData.error
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      
      return data

    } catch(err) {
      throw err
    }
  }

  export const get_user = async () => {
    try {
      const userData = await api_request(
        'auth/@me',
        'GET'
      )

      return userData

    } catch (err) {
      console.log(err)
    }
  }
