import React from 'react'
import {useState} from 'react';
import { faEye, faEyeSlash } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'

function PwdInput({onChange, name}) {

    const [showPwd, setShowPwd] = useState(false)
    
    return (
        <div className='flex items-center'>
            <input type={`${showPwd? 'text' : 'password'}`} autoComplete='off' autoFocus='off' 
                            placeholder={name}
                            className='p-2 rounded' 
                            onInput={(e) => onChange(e.target.value)}
             />
            <FontAwesomeIcon onClick={() => setShowPwd((h) => !h)} 
                className='w-4 relative right-6 hover:cursor-pointer' 
                icon={showPwd ? faEye : faEyeSlash} 
                color={'#00A881'}
            />
        </div>
        
    )
}

export default PwdInput