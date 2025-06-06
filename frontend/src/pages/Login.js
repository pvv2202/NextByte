import React, { useState } from 'react'
import { useForm } from 'react-hook-form';
import { api_request } from '../api_request';
import LoginForm from '../components/loginForm';
import { useNavigate} from 'react-router';

function Login() {
    const navigate = useNavigate()
    const [authenticating, setAuthenticating] = useState(false)
    const {register, handleSubmit, reset} = useForm() 
    const [userError, setUserError] = useState('')
    const [pwdError, setPwdError] = useState('')
    
    const onSubmit = async (data) => {
        setUserError('')
        setPwdError('')
        setAuthenticating(true)
        try{
            const response = await api_request(
                'login',
                'POST',
                {'Content-Type': 'application/json'},
                {username: data['username'], password: data['password']}
            )
            // TODO: SET UP USER SESSION?
            navigate('/workspace')
            
        } catch(err){
            if(err.message.includes('User')) setUserError(err.message)
            else setPwdError(err.message)
        }
        
        setAuthenticating(false)
        
    }
    return (
    <div className='w-full h-screen flex justify-center items-center'>
        <LoginForm
            navigate={navigate}
            authenticating={authenticating}
            register={register} handleSubmit={handleSubmit} reset={reset}
            userError={userError} 
            pwdError={pwdError} 
            onSubmit={onSubmit}  
        />
    </div>
    
    )
}

export default Login