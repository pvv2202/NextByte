import React from 'react'
import { useNavigate } from 'react-router'
import { useState, useEffect, useContext } from 'react'
import { useForm, Controller } from 'react-hook-form'
import SignupForm from '../components/auth/signupForm'
import { api_request } from '../api_request'
import { get_user } from '../api_request'
import { UserContext } from '../UserContext'

function SignUp() {
  const navigate = useNavigate()
  const [authenticating, setAuthenticating] = useState(false)
  const {register, handleSubmit, reset, control, watch} = useForm() 
  const [userError, setUserError] = useState('')
  const {user, setUser} = useContext(UserContext)

  useEffect(() => {
          if (user) navigate('/landing')
  }, [])
  
  const onSubmit = async (data) => {
      setUserError('')
      setAuthenticating(true)
      try{
          const response = await api_request(
              'signup',
              'POST',
              {'Content-Type': 'application/json'},
              {username: data['username'], 
                password: data['password'],
                age: data['age'],
                email: data['email'],
                country: data['country'],
                state: data['state'],
                city: data['city']

            }
          )
          navigate('/login')
          
      } catch(err){
          setUserError(err.message)
          
      }
      
      setAuthenticating(false)
      
  }
  return (
    <>
    <div className='w-full h-screen flex justify-center items-center gap-2'>
        <SignupForm
            navigate={navigate}
            authenticating={authenticating}
            register={register} handleSubmit={handleSubmit} reset={reset}
            userError={userError}
            onSubmit={onSubmit}  
            Controller={Controller}
            control={control}
            watch={watch}
        />
        
    </div>
    
    </>
  )
}

export default SignUp