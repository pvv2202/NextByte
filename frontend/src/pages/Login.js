import React, { useEffect, useState, useContext } from 'react'
import { useForm } from 'react-hook-form';
import { api_request } from '../api_request';
import LoginForm from '../components/auth/loginForm';
import { useNavigate} from 'react-router';
import { get_user } from '../api_request';
import { UserContext } from '../UserContext';


function Login() {
    const navigate = useNavigate()
    const [authenticating, setAuthenticating] = useState(false)
    const {register, handleSubmit, reset} = useForm() 
    const [userError, setUserError] = useState('')
    const [pwdError, setPwdError] = useState('')
    const {user, setUser} = useContext(UserContext)
    
    useEffect(() => {
        if (user) navigate('/landing')
    }, [])
    


    const onSubmit = async (data) => {
        setUserError('')
        setPwdError('')
        setAuthenticating(true)
        try{
            const authUserDetails = await api_request(
                'auth/login',
                'POST',
                {'Content-Type': 'application/json'},
                {username: data['username'], password: data['password']}
            )
            setUser(authUserDetails)
            navigate('/landing')
            
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