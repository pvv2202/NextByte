import React from 'react'
import { useForm } from 'react-hook-form';
import { api_request } from '../api_request';
import { PulseDiv } from '../components/bot';

function Login() {
    const {register, handleSubmit, reset} = useForm() 
    const onSubmit = async (data) => {
        const response = await api_request(
            'login',
            'POST',
            {'Content-Type': 'application/json'},
            {username: data['username'], password: data['password']}
        )
        
    }
    return (
    <div className='w-full h-screen flex justify-center items-center'>
        <form className="flex flex-col gap-4 bg-white p-6 rounded"onSubmit={handleSubmit(onSubmit)}>
                    <h2 className='flex gap-1 items-center justify-center text-gray-800 text-xl mt-4'>
                        Welcome Back
                        <PulseDiv>
                            <img className=' max-w-12 object-contain' src={'Closing-eyes.gif'} alt="nextbyte" />
                        </PulseDiv>
                    </h2>
                    <input type='text' {...register("username", {required: true,
                        maxLength: 20})} 
                        placeholder='Username'
                        className='p-2 rounded'
                    />
                    <input type="text" {...register("password", {required: true, 
                        maxLength:20})}
                        placeholder='Password'
                        className='p-2 rounded' 
                    />
                    <button type='submit' className='rounded curser:hand hover:scale-105 text-white  ml-3 p-2 bg-gray-800'>Login</button>
            </form>
    </div>
    
    )
}

export default Login