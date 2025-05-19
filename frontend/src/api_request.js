
// TODO: change upon publishing
const API_URL = 'http://127.0.0.1:5000/api'

export const api_request = async (endpoint, method, headers, body) => {
    try{
      const response = await fetch(`${API_URL}/${endpoint}`, {
        method: method,
        headers: {...headers},
        body: JSON.stringify(body)
      })

      if (!response.ok) throw new Error('error fetching data')
      const data = await response.json();
      
      return data

    } catch(err) {
      console.log(`Error during fetch: ${err}`)
    }
  }
