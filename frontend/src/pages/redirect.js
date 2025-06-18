import React from 'react'
import { useEffect } from 'react'
import { useNavigate } from 'react-router'
import { ClipLoader } from 'react-spinners'

function Redirect({redirect, setRedirect}) {
    const navigate = useNavigate()
    
    let message = ''
    let path = ''
    if (redirect){
        message = redirect['message']
        path = redirect['path']
    }
    
    useEffect(()=>{
        setTimeout(() => {
            setRedirect(null)
            navigate(path)   
        }, 2000)
    })
    return (
    <div> 
        <p>{message}</p>
        <ClipLoader color={'black'} speedMultiplier={0.5} size={20} />
    </div>
    )
}

export default Redirect